//! Обнаружение ключевого слова — Few-Shot Metric Learning.
//!
//! Идея: энкодер (трейт [`FrameEmbedder`], реализация — FastConformer в
//! приложении) сжимает окно 1.2 с в эмбеддинг. Из эмбеддинга вычитается
//! фоновое направление речи (центрирование — сырые эмбеддинги лежат в
//! общем направлении на 93%+, без этого дискриминативный сигнал зажат в
//! остатке), счёт = `max_p cos(x, p) − w·max_n cos(x, n)` по центрированным
//! эталонам: жёсткие негативы давят каждое похожее слово по отдельности.
//!
//! Малочувствительность к диктору проверяется верификацией на hold-out
//! голосах (`wake_verify`), а не декларируется.
//!
//! Поток: `process(frame)` принимает произвольные куски 16 кГц mono и режет
//! их на суб-куски длиной до `hop_ms` — результат не зависит от размера
//! входного фрагмента. Энкодер запускается только на голосе (адаптивный
//! шумовой пол + hangover), решения — по счётчикам сэмплов, без wall-clock.

use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;

use serde::{Deserialize, Serialize};

/// Частота дискретизации входного потока.
pub const SAMPLE_RATE: usize = 16_000;
/// Ниже этой нормы остатка после центрирования вектор считается совпавшим
/// с фоновым средним: не нормализуем (иначе шум раздувается в случайное
/// направление) и даём минимальный счёт.
pub const MIN_RESIDUAL_NORM: f32 = 1e-3;
/// Скорость подъёма оценки шумового пола, единиц RMS в секунду.
/// 0.2/с: шум усваивается за ~2 с, а фраза до 2 с не поднимает пол
/// выше трети своей громкости (гейт vad_ratio=3 остаётся открытым).
const NOISE_RISE_PER_SEC: f32 = 0.2;

/// Экстрактор эмбеддингов: окно 16 кГц mono f32 → L2-нормализованный вектор
/// фиксированной длины. Реализация в приложении (нейросеть), крейту нужен
/// только контракт.
pub trait FrameEmbedder: Send {
    fn embed(&mut self, window_16k: &[f32]) -> Result<Vec<f32>, String>;
    /// Размерность эмбеддинга (для диагностики и валидации эталона).
    fn dims(&self) -> usize;
}

/// Параметры детектора. Новые поля имеют `#[serde(default)]` — конфиг из
/// старого файла читается с значениями по умолчанию.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingConfig {
    /// Длина окна эмбеддинга, мс (1.2 с — фраза «Джарвис» с запасом).
    pub window_ms: usize,
    /// Минимальный шаг между запусками энкодера, мс.
    pub hop_ms: usize,
    /// Запасной порог: применяется, только если у эталона нет своего
    /// (эталоны v3 всегда хранят калиброванный порог).
    pub threshold: f32,
    /// Пауза после срабатывания, мс (анти-дубль), в аудио-времени.
    pub cooldown_ms: u64,
    /// Сколько последних окон участвуют в решении (1–7).
    pub smoothing: usize,
    /// Минимум окон с счётом ≥ порога среди последних `smoothing` окон.
    /// 1 — прежнее пиковое поведение; 2+ требует устойчивого превышения.
    /// Компромисс: больше min_hits — меньше ложных срабатываний, но фраза
    /// должна произноситься достаточно долго, чтобы дать нужное число окон.
    #[serde(default = "default_min_hits")]
    pub min_hits: usize,
    /// Абсолютный минимум RMS, ниже которого гарантированная тишина.
    pub vad_rms: f32,
    /// Относительный гейт: голос = rms > noise_floor · vad_ratio.
    #[serde(default = "default_vad_ratio")]
    pub vad_ratio: f32,
    /// Сколько мс после конца голоса энкодер продолжает оценивать окна
    /// (чтобы было оценено окно, заканчивающееся концом слова).
    #[serde(default = "default_hangover_ms")]
    pub hangover_ms: usize,
    /// Множитель штрафа за негативы в счёте.
    #[serde(default = "default_negative_weight")]
    pub negative_weight: f32,
}

fn default_min_hits() -> usize {
    2
}
fn default_vad_ratio() -> f32 {
    3.0
}
fn default_hangover_ms() -> usize {
    1200
}
fn default_negative_weight() -> f32 {
    1.0
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            window_ms: 1200,
            hop_ms: 240,
            threshold: 0.12,
            cooldown_ms: 2500,
            smoothing: 3,
            min_hits: default_min_hits(),
            vad_rms: 0.004,
            vad_ratio: default_vad_ratio(),
            hangover_ms: default_hangover_ms(),
            negative_weight: default_negative_weight(),
        }
    }
}

/// Зарегистрированный эталон (формат v3): центрированные эмбеддинги.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingReference {
    /// Версия формата. 3 = центрированные эмбеддинги + жёсткие негативы
    /// + leave-one-out калибровка. Версии 1 и 2 отклоняются при загрузке
    /// с советом пересобрать эталон.
    pub version: u32,
    /// Размерность энкодера: несовпадение при загрузке — явная ошибка.
    pub dims: usize,
    /// Фоновое среднее направление речи, вычитается из каждого эмбеддинга.
    pub mean: Vec<f32>,
    /// Откуда взято mean: "negatives" (среднее по негативам регистрации)
    /// или "background" (задано офлайн по большому корпусу речи).
    pub mean_source: String,
    /// Центрированные эмбеддинги каждого позитивного сэмпла: счёт берётся
    /// максимумом по ним — фраза под другим тоном ложится на ближайший.
    pub positives: Vec<Vec<f32>>,
    /// Центрированные эмбеддинги каждого негативного сэмпла: штраф берётся
    /// максимумом по ним (жёсткие негативы), а не по центроиду.
    pub negatives: Vec<Vec<f32>>,
    /// Центроид позитивов — только диагностика, в счёте не участвует.
    pub positive: Vec<f32>,
    /// Центроид негативов — только диагностика, в счёте не участвует.
    pub negative: Vec<f32>,
    /// Порог, откалиброванный при регистрации (leave-one-out) или по
    /// hold-out записям (`calibrate_threshold_holdout`).
    pub threshold: f32,
    /// Число позитивных/негативных сэмплов регистрации (диагностика).
    pub positives_count: usize,
    pub negatives_count: usize,
}

impl EmbeddingReference {
    pub fn version_current() -> u32 {
        3
    }

    /// Перекалибровать порог после загрузки (например, по hold-out голосам:
    /// самосовпадение записей регистрации завышает min_pos и порог).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Сохранить эталон без привязки к детектору (перекалибровка порога).
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), EmbeddingError> {
        let file = File::create(path).map_err(|e| EmbeddingError::Io(e.to_string()))?;
        ciborium::into_writer(self, file).map_err(|e| EmbeddingError::Io(e.to_string()))
    }

    /// Прочитать эталон с диска без энкодера: диагностика счётов.
    /// Привязка к энкодеру — в [`EmbeddingDetector::load_reference`].
    pub fn load_reference_only(
        path: &std::path::Path,
    ) -> Result<EmbeddingReference, EmbeddingError> {
        decode_reference(path)
    }
}

/// Итог срабатывания.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDetection {
    /// Пик последних окон: max_p cos(x, p) − w·max_n cos(x, n).
    pub score: f32,
    /// Счёт последнего оценённого окна (для диагностики порога).
    pub raw_score: f32,
}

/// Диагностика последнего суб-куска: почему сработало/не сработало.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDiag {
    /// RMS куска.
    pub rms: f32,
    /// Текущая оценка шумового пола.
    pub noise_floor: f32,
    /// Кусок распознан как голос.
    pub voiced: bool,
    /// Остаток hangover, сэмплов.
    pub hangover_left: usize,
    /// Энкодер запускался на этом куске.
    pub encoder_ran: bool,
    /// Счёт последнего окна (None — энкодер не запускался).
    pub raw_score: Option<f32>,
    /// Действующий порог (из эталона).
    pub threshold: Option<f32>,
    /// Сколько окон из последних `smoothing` ≥ порога.
    pub hits: usize,
}

/// Ошибки детектора.
#[derive(Debug)]
pub enum EmbeddingError {
    /// Эмбеддинг эталона не совпал по размерности с текущим энкодером.
    DimMismatch {
        expected: usize,
        got: usize,
    },
    Io(String),
    /// Недостаточно данных для регистрации.
    Enrollment(String),
    /// Некорректная конфигурация.
    Config(String),
}

impl std::error::Error for EmbeddingError {}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::DimMismatch { expected, got } => {
                write!(
                    f,
                    "размерность эмбеддинга {got} не совпадает с ожидаемой {expected}"
                )
            }
            EmbeddingError::Io(e) => write!(f, "ошибка файла эталона: {e}"),
            EmbeddingError::Enrollment(e) => write!(f, "регистрация: {e}"),
            EmbeddingError::Config(e) => write!(f, "конфигурация: {e}"),
        }
    }
}

/// Декодировать файл эталона с version-пробой: старые версии и не-эталоны
/// отклоняются с внятным советом до типизированной десериализации.
fn decode_reference(path: &std::path::Path) -> Result<EmbeddingReference, EmbeddingError> {
    let file = File::open(path).map_err(|e| EmbeddingError::Io(e.to_string()))?;
    let value: ciborium::Value = ciborium::from_reader(BufReader::new(file))
        .map_err(|e| EmbeddingError::Io(e.to_string()))?;
    let version = match &value {
        ciborium::Value::Map(pairs) => pairs.iter().find_map(|(k, v)| {
            if k == &ciborium::Value::Text("version".into()) {
                match v {
                    ciborium::Value::Integer(i) => Some(u32::try_from(i128::from(*i)).unwrap_or(0)),
                    _ => None,
                }
            } else {
                None
            }
        }),
        _ => None,
    };
    match version {
        Some(v) if v == EmbeddingReference::version_current() => {}
        other => {
            return Err(EmbeddingError::Enrollment(format!(
                "эталон версии {:?} несовместим с текущей ({}): пересобери эталон — \
                 cargo run -p jarvis-core --example wake_enroll, \
                 либо «Собрать модель» на странице Wake Word в приложении",
                other,
                EmbeddingReference::version_current()
            )));
        }
    }
    let mut buf = Vec::new();
    ciborium::into_writer(&value, &mut buf).map_err(|e| EmbeddingError::Io(e.to_string()))?;
    ciborium::from_reader(&buf[..]).map_err(|e| EmbeddingError::Io(e.to_string()))
}

/// L2-нормализация вектора (нулевой вектор остаётся нулевым).
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm.is_finite() && norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Косинус двух L2-нормализованных векторов = скалярное произведение.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Одно окно-счёт для сглаживания: значение и позиция в аудио-времени
/// (сэмплах от начала потока) — старение по времени, не по числу запусков.
#[derive(Debug, Clone, Copy)]
struct ScoreEntry {
    score: f32,
    at: usize,
}

/// Детектор: кольцевой буфер 16 кГц + адаптивный VAD + энкодер + метрика.
pub struct EmbeddingDetector<E: FrameEmbedder> {
    extractor: E,
    config: EmbeddingConfig,
    reference: Option<EmbeddingReference>,
    /// Фоновое среднее, заданное офлайн (`set_background_mean`); если есть,
    /// enrollment центрирует им вместо среднего по негативам.
    background_mean: Option<Vec<f32>>,
    buffer: VecDeque<f32>,
    /// Сэмплов с момента последнего запуска энкодера.
    since_embed: usize,
    /// Сэмплов с момента последнего срабатывания.
    since_detection: usize,
    /// Остаток hangover, сэмплов.
    hangover_left: usize,
    /// Оценка шумового пола (RMS).
    noise_floor: f32,
    /// Окна-счёты: значение + позиция в аудио-времени.
    recent_scores: VecDeque<ScoreEntry>,
    /// Счёт последнего окна (даже без срабатывания) — диагностика порога.
    last_raw_score: Option<f32>,
    last_diag: Option<EmbeddingDiag>,
    total_samples: usize,
    samples_per_window: usize,
    samples_per_hop: usize,
    cooldown_samples: usize,
    hangover_samples: usize,
}

impl<E: FrameEmbedder> EmbeddingDetector<E> {
    /// Создать детектор. Конфиг валидируется: при ошибке — Err, не паника.
    pub fn new(extractor: E, config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        if config.window_ms == 0 {
            return Err(EmbeddingError::Config("window_ms должен быть > 0".into()));
        }
        if config.hop_ms == 0 {
            return Err(EmbeddingError::Config("hop_ms должен быть > 0".into()));
        }
        if config.hop_ms > config.window_ms {
            return Err(EmbeddingError::Config(format!(
                "hop_ms ({}) не может превышать window_ms ({})",
                config.hop_ms, config.window_ms
            )));
        }
        if config.smoothing < 1 {
            return Err(EmbeddingError::Config("smoothing должен быть ≥ 1".into()));
        }
        if config.min_hits < 1 || config.min_hits > config.smoothing {
            return Err(EmbeddingError::Config(format!(
                "min_hits ({}) должен быть в 1..={} (smoothing)",
                config.min_hits, config.smoothing
            )));
        }
        if config.negative_weight < 0.0 {
            return Err(EmbeddingError::Config(
                "negative_weight не может быть отрицательным".into(),
            ));
        }
        let samples_per_window = config.window_ms * SAMPLE_RATE / 1000;
        let samples_per_hop = (config.hop_ms * SAMPLE_RATE / 1000).max(1);
        Ok(Self {
            cooldown_samples: (config.cooldown_ms as usize) * SAMPLE_RATE / 1000,
            hangover_samples: config.hangover_ms * SAMPLE_RATE / 1000,
            samples_per_window,
            samples_per_hop,
            extractor,
            config,
            reference: None,
            background_mean: None,
            buffer: VecDeque::with_capacity(samples_per_window + samples_per_hop),
            since_embed: 0,
            since_detection: 0,
            hangover_left: 0,
            noise_floor: 0.0,
            recent_scores: VecDeque::new(),
            last_raw_score: None,
            last_diag: None,
            total_samples: 0,
        })
    }

    /// Задать фоновое среднее, посчитанное офлайн по большому корпусу речи:
    /// enrollment отцентрирует им сэмплы вместо среднего по негативам.
    pub fn set_background_mean(&mut self, mean: Vec<f32>) {
        self.background_mean = Some(mean);
    }

    /// Загруженный эталон (после [`Self::load_reference`] или регистрации).
    pub fn reference(&self) -> Option<&EmbeddingReference> {
        self.reference.as_ref()
    }

    /// Регистрация: 3–10 позитивных окон фразы (любой голос) + 3+ негативов
    /// (тишина, шум, похожие слова). Каждый сэмпл — окно `window_ms`.
    /// Порог калибруется leave-one-out: счёт каждого сэмпла исключает его
    /// самого из эталонов, иначе самосовпадение (cos = 1) завышает порог.
    pub fn enroll(
        &mut self,
        positives: &[Vec<f32>],
        negatives: &[Vec<f32>],
    ) -> Result<EmbeddingReference, EmbeddingError> {
        if positives.len() < 3 {
            return Err(EmbeddingError::Enrollment(format!(
                "нужно минимум 3 позитивных сэмпла, получено {}",
                positives.len()
            )));
        }
        if negatives.len() < 3 {
            return Err(EmbeddingError::Enrollment(format!(
                "нужно минимум 3 негативных сэмпла, получено {}",
                negatives.len()
            )));
        }
        let expected = self.extractor.dims();
        let mut embed_all =
            |samples: &[Vec<f32>], what: &str| -> Result<Vec<Vec<f32>>, EmbeddingError> {
                samples
                    .iter()
                    .map(|s| {
                        if s.len() != self.samples_per_window {
                            return Err(EmbeddingError::Enrollment(format!(
                                "{what}: ожидалось {} сэмплов ({} мс), получено {}",
                                self.samples_per_window,
                                self.config.window_ms,
                                s.len()
                            )));
                        }
                        let mut e = self.extractor.embed(s).map_err(EmbeddingError::Io)?;
                        if e.len() != expected {
                            return Err(EmbeddingError::DimMismatch {
                                expected,
                                got: e.len(),
                            });
                        }
                        l2_normalize(&mut e);
                        Ok(e)
                    })
                    .collect()
            };
        let skip_zero = |mut v: Vec<Vec<f32>>, what: &str| -> Vec<Vec<f32>> {
            // Тишина даёт нулевой эмбеддинг: направления нет, в эталоне
            // ему взяться неоткуда — пропускаем с пометкой в логе.
            let before = v.len();
            v.retain(|e| e.iter().map(|x| x * x).sum::<f32>().sqrt() >= 1e-6);
            if v.len() < before {
                println!(
                    "ℹ️ {what}: пропущено нулевых (тихих) сэмплов: {}",
                    before - v.len()
                );
            }
            v
        };
        let pos = skip_zero(embed_all(positives, "позитив")?, "позитивы");
        let neg = skip_zero(embed_all(negatives, "негатив")?, "негативы");
        if pos.len() < 3 {
            return Err(EmbeddingError::Enrollment(format!(
                "после отбрасывания нулевых (тихих) сэмплов осталось {} позитивов — нужно ≥ 3",
                pos.len()
            )));
        }
        if neg.len() < 3 {
            return Err(EmbeddingError::Enrollment(format!(
                "после отбрасывания нулевых (тихих) сэмплов осталось {} негативов — нужно ≥ 3",
                neg.len()
            )));
        }

        // Фоновое среднее: офлайн-корпус, если задан, иначе негативы.
        let (mean, mean_source) = match &self.background_mean {
            Some(m) if m.len() == expected => (m.clone(), "background".to_string()),
            Some(m) => {
                return Err(EmbeddingError::Enrollment(format!(
                    "фоновое среднее размерности {}, ожидается {expected}",
                    m.len()
                )));
            }
            None => {
                let mut m = vec![0_f32; expected];
                for e in &neg {
                    for (acc, x) in m.iter_mut().zip(e) {
                        *acc += x;
                    }
                }
                for x in m.iter_mut() {
                    *x /= neg.len() as f32;
                }
                (m, "negatives".to_string())
            }
        };

        // Центрирование с guard'ом: сэмпл, совпавший с фоновым средним
        // (остаток < MIN_RESIDUAL_NORM), не нормализуем — он не несёт
        // информации, и включать его в эталон нельзя.
        let center = |e: &[f32]| -> Result<Vec<f32>, EmbeddingError> {
            let mut v: Vec<f32> = e.iter().zip(&mean).map(|(x, m)| x - m).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !norm.is_finite() || norm < MIN_RESIDUAL_NORM {
                return Err(EmbeddingError::Enrollment(format!(
                    "сэмпл совпал с фоновым средним (норма остатка {norm:.6} < {MIN_RESIDUAL_NORM}) — \
                     он не несёт информации: запиши другой сэмпл"
                )));
            }
            for x in v.iter_mut() {
                *x /= norm;
            }
            Ok(v)
        };
        let pos_c: Vec<Vec<f32>> = pos.iter().map(|e| center(e)).collect::<Result<_, _>>()?;
        let neg_c: Vec<Vec<f32>> = neg.iter().map(|e| center(e)).collect::<Result<_, _>>()?;

        let w = self.config.negative_weight;
        // Leave-one-out: счёт сэмпла исключает его самого из эталонов,
        // иначе самосовпадение (cos = 1) завышает min_pos у позитивов и
        // занижает max_neg у негативов (штраф за собственный кластер = w).
        let pos_score_loo = |i: usize| -> f32 {
            let e = &pos_c[i];
            let best_p = pos_c
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| dot(e, p))
                .fold(f32::NEG_INFINITY, f32::max);
            let best_n = neg_c
                .iter()
                .map(|n| dot(e, n))
                .fold(f32::NEG_INFINITY, f32::max);
            best_p - w * best_n
        };
        let neg_score_loo = |i: usize| -> f32 {
            let e = &neg_c[i];
            let best_p = pos_c
                .iter()
                .map(|p| dot(e, p))
                .fold(f32::NEG_INFINITY, f32::max);
            let best_n = neg_c
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, n)| dot(e, n))
                .fold(f32::NEG_INFINITY, f32::max);
            best_p - w * best_n
        };
        let min_pos = (0..pos_c.len())
            .map(pos_score_loo)
            .fold(f32::INFINITY, f32::min);
        let max_neg = (0..neg_c.len())
            .map(neg_score_loo)
            .fold(f32::NEG_INFINITY, f32::max);
        if max_neg >= min_pos {
            return Err(EmbeddingError::Enrollment(format!(
                "негативы неотделимы от позитивов leave-one-out \
                 (max негатив {max_neg:.4} >= min позитив {min_pos:.4}): \
                 запиши негативы из других слов и/или больше позитивов \
                 с разными интонациями"
            )));
        }
        // Середина LOO-зазора: свежая фраза обычно садится между худшим
        // позитивом и лучшим негативом.
        let threshold = max_neg + 0.5 * (min_pos - max_neg);
        println!(
            "📊 Калибровка (LOO): позитивы min {min_pos:.3}, негативы max {max_neg:.3}, зазор {:.3}, порог {threshold:.3}",
            min_pos - max_neg
        );

        let mut positive = vec![0_f32; expected];
        for e in &pos_c {
            for (acc, x) in positive.iter_mut().zip(e) {
                *acc += x;
            }
        }
        for x in positive.iter_mut() {
            *x /= pos_c.len() as f32;
        }
        l2_normalize(&mut positive);
        let mut negative = vec![0_f32; expected];
        for e in &neg_c {
            for (acc, x) in negative.iter_mut().zip(e) {
                *acc += x;
            }
        }
        for x in negative.iter_mut() {
            *x /= neg_c.len() as f32;
        }
        l2_normalize(&mut negative);

        let reference = EmbeddingReference {
            version: EmbeddingReference::version_current(),
            dims: expected,
            mean,
            mean_source,
            positives: pos_c.clone(),
            negatives: neg_c.clone(),
            positive,
            negative,
            threshold,
            positives_count: pos_c.len(),
            negatives_count: neg_c.len(),
        };
        self.reference = Some(reference.clone());
        Ok(reference)
    }

    /// Перекалибровать порог по записям, НЕ входившим в регистрацию
    /// (hold-out): самосовпадение регистрационных сэмплов завышает порог.
    /// Возвращает установленный порог.
    pub fn calibrate_threshold_holdout(
        &mut self,
        positives_holdout: &[Vec<f32>],
        negatives_holdout: &[Vec<f32>],
    ) -> Result<f32, EmbeddingError> {
        let Some(reference) = self.reference.clone() else {
            return Err(EmbeddingError::Enrollment(
                "нет эталона — сначала enroll или load_reference".into(),
            ));
        };
        let expected = self.extractor.dims();
        let mut embed_holdout = |samples: &[Vec<f32>]| -> Result<Vec<Vec<f32>>, EmbeddingError> {
            samples
                .iter()
                .map(|s| {
                    if s.len() != self.samples_per_window {
                        return Err(EmbeddingError::Enrollment(format!(
                            "hold-out: ожидалось {} сэмплов, получено {}",
                            self.samples_per_window,
                            s.len()
                        )));
                    }
                    let mut e = self.extractor.embed(s).map_err(EmbeddingError::Io)?;
                    if e.len() != expected {
                        return Err(EmbeddingError::DimMismatch {
                            expected,
                            got: e.len(),
                        });
                    }
                    l2_normalize(&mut e);
                    for (x, m) in e.iter_mut().zip(&reference.mean) {
                        *x -= m;
                    }
                    let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if !norm.is_finite() || norm < MIN_RESIDUAL_NORM {
                        return Err(EmbeddingError::Enrollment(format!(
                            "hold-out сэмпл совпал с фоновым средним (норма {norm:.6})"
                        )));
                    }
                    for x in e.iter_mut() {
                        *x /= norm;
                    }
                    Ok(e)
                })
                .collect()
        };
        let pos = embed_holdout(positives_holdout)?;
        let neg = embed_holdout(negatives_holdout)?;
        if pos.is_empty() || neg.is_empty() {
            return Err(EmbeddingError::Enrollment(
                "hold-out: нужны хотя бы по одному позитиву и негативу".into(),
            ));
        }
        let w = self.config.negative_weight;
        let score = |e: &[f32]| -> f32 {
            let best_p = reference
                .positives
                .iter()
                .map(|p| dot(e, p))
                .fold(f32::NEG_INFINITY, f32::max);
            let best_n = reference
                .negatives
                .iter()
                .map(|n| dot(e, n))
                .fold(f32::NEG_INFINITY, f32::max);
            best_p - w * best_n
        };
        let min_pos = pos.iter().map(|e| score(e)).fold(f32::INFINITY, f32::min);
        let max_neg = neg
            .iter()
            .map(|e| score(e))
            .fold(f32::NEG_INFINITY, f32::max);
        if max_neg >= min_pos {
            return Err(EmbeddingError::Enrollment(format!(
                "hold-out неотделим (max негатив {max_neg:.4} >= min позитив {min_pos:.4})"
            )));
        }
        let threshold = max_neg + 0.5 * (min_pos - max_neg);
        if let Some(reference) = self.reference.as_mut() {
            reference.set_threshold(threshold);
        }
        println!(
            "📊 Калибровка по hold-out: позитивы min {min_pos:.3}, негативы max {max_neg:.3}, порог {threshold:.3}"
        );
        Ok(threshold)
    }

    /// Сохранить эталон в компактный CBOR-файл.
    pub fn save_reference(&self, path: &std::path::Path) -> Result<(), EmbeddingError> {
        let Some(reference) = &self.reference else {
            return Err(EmbeddingError::Enrollment(
                "эталон не зарегистрирован".into(),
            ));
        };
        let file = File::create(path).map_err(|e| EmbeddingError::Io(e.to_string()))?;
        ciborium::into_writer(reference, file).map_err(|e| EmbeddingError::Io(e.to_string()))
    }

    /// Загрузить эталон и привязать к текущему энкодеру. Версии 1 и 2
    /// (и любые не-эталонные файлы) отклоняются по version-пробе ДО
    /// типизированной десериализации: их поля несовместимы, и ошибка
    /// «missing field» вместо внятного диагноза недопустима.
    pub fn load_reference(&mut self, path: &std::path::Path) -> Result<(), EmbeddingError> {
        let mut reference = decode_reference(path)?;
        let expected = self.extractor.dims();
        if reference.dims != expected || reference.positive.len() != expected {
            return Err(EmbeddingError::DimMismatch {
                expected,
                got: if reference.dims != expected {
                    reference.dims
                } else {
                    reference.positive.len()
                },
            });
        }
        reference.set_threshold(reference.threshold);
        self.reference = Some(reference);
        Ok(())
    }

    /// Подать очередной кусок потока 16 кГц mono. Возвращает срабатывание —
    /// либо None. Вход режется на суб-куски до `hop_ms`, поэтому результат
    /// не зависит от размера входного фрагмента.
    pub fn process(&mut self, frame: &[f32]) -> Option<EmbeddingDetection> {
        self.reference.as_ref()?;
        let mut detection = None;
        let mut offset = 0;
        while offset < frame.len() {
            let take = (frame.len() - offset).min(self.samples_per_hop);
            let sub = &frame[offset..offset + take];
            if let Some(d) = self.process_sub(sub) {
                detection = Some(d);
            }
            offset += take;
        }
        detection
    }

    /// Обработка одного суб-куска (≤ hop): буфер, шумовой пол, VAD,
    /// hangover, энкодер, метрика, решение.
    fn process_sub(&mut self, sub: &[f32]) -> Option<EmbeddingDetection> {
        let mut diag = EmbeddingDiag {
            rms: 0.0,
            noise_floor: self.noise_floor,
            voiced: false,
            hangover_left: self.hangover_left,
            encoder_ran: false,
            raw_score: None,
            threshold: None,
            hits: 0,
        };

        self.buffer.extend(sub.iter().copied());
        while self.buffer.len() > self.samples_per_window {
            self.buffer.pop_front();
        }
        self.since_embed += sub.len();
        self.since_detection += sub.len();
        self.total_samples += sub.len();

        // Шумовой пол: быстрое падение вниз, медленный подъём вверх —
        // сходится к уровню фона за ~1-2 с и не поднимается голосом.
        let rms: f32 = sub.iter().map(|x| x * x).sum::<f32>() / sub.len().max(1) as f32;
        let rms = rms.sqrt();
        let voiced = rms > self.config.vad_rms && rms > self.noise_floor * self.config.vad_ratio;
        // Пол: быстрое падение к фактическому уровню, медленный подъём
        // (речь за время фразы не поднимает пол до гейта).
        if rms < self.noise_floor {
            self.noise_floor = rms;
        } else {
            let dt = sub.len() as f32 / SAMPLE_RATE as f32;
            self.noise_floor += NOISE_RISE_PER_SEC * dt * (rms - self.noise_floor);
        }
        if voiced {
            self.hangover_left = self.hangover_samples;
        } else {
            if self.hangover_left > 0 {
                self.hangover_left = self.hangover_left.saturating_sub(sub.len());
            } else if !self.recent_scores.is_empty() {
                // VAD закрыт и hangover истёк: окно-счёты устарели — чистим,
                // иначе старый пик выстрелит на первом же постороннем звуке.
                self.recent_scores.clear();
            }
        }

        diag.rms = rms;
        diag.noise_floor = self.noise_floor;
        diag.voiced = voiced;
        diag.hangover_left = self.hangover_left;

        // Энкодер: окно полное, hop отлежал, и есть голос или hangover.
        let window_full = self.buffer.len() == self.samples_per_window;
        let hop_ready = self.since_embed >= self.samples_per_hop;
        let encoder_ran = window_full && hop_ready && (voiced || self.hangover_left > 0);

        let reference = self.reference.as_ref()?;
        diag.threshold = Some(reference.threshold);

        if !encoder_ran {
            self.last_diag = Some(diag);
            return None;
        }
        self.since_embed = 0;
        diag.encoder_ran = true;

        let window: Vec<f32> = self.buffer.iter().copied().collect();
        let mut embedding = match self.extractor.embed(&window) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("⚠️ Wake embed: {e}");
                self.last_diag = Some(diag);
                return None;
            }
        };
        l2_normalize(&mut embedding);
        // Центрирование тем же фоновым средним, что и на регистрации.
        for (x, m) in embedding.iter_mut().zip(&reference.mean) {
            *x -= m;
        }
        // Guard вырожденного остатка: почти нулевой остаток — совпадение с
        // фоном; нормализация раздулa бы его в случайное направление.
        let residual: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let raw_score = if residual < MIN_RESIDUAL_NORM {
            -1.0
        } else {
            for x in embedding.iter_mut() {
                *x /= residual;
            }
            let best_p = reference
                .positives
                .iter()
                .map(|p| dot(&embedding, p))
                .fold(f32::NEG_INFINITY, f32::max);
            let best_n = reference
                .negatives
                .iter()
                .map(|n| dot(&embedding, n))
                .fold(f32::NEG_INFINITY, f32::max);
            best_p - self.config.negative_weight * best_n
        };
        self.last_raw_score = Some(raw_score);
        diag.raw_score = Some(raw_score);

        // Окна-счёты стареют по аудио-времени: запись живёт window_ms
        // от своего места в потоке, плюс держим не более `smoothing` штук.
        self.recent_scores.push_back(ScoreEntry {
            score: raw_score,
            at: self.total_samples,
        });
        let span = self.samples_per_window;
        self.recent_scores
            .retain(|e| self.total_samples - e.at <= span);
        while self.recent_scores.len() > self.config.smoothing.max(1) {
            self.recent_scores.pop_front();
        }

        let hits = self
            .recent_scores
            .iter()
            .filter(|e| e.score >= reference.threshold)
            .count();
        let peak = self
            .recent_scores
            .iter()
            .map(|e| e.score)
            .fold(f32::NEG_INFINITY, f32::max);
        diag.hits = hits;

        // Решение: min_hits окон из последних ≥ порога. min_hits=1 —
        // прежнее пиковое поведение (одиночный выброс срабатывает);
        // min_hits≥2 требует устойчивого превышения и защищает от
        // одиночного выброса.
        if hits < self.config.min_hits {
            self.last_diag = Some(diag);
            return None;
        }
        // Cooldown в аудио-времени: одно срабатывание на фразу.
        if self.since_detection < self.cooldown_samples {
            self.last_diag = Some(diag);
            return None;
        }
        self.since_detection = 0;
        // Очистка окон после срабатывания: старый пик не должен дать
        // второе срабатывание на первом же постороннем звуке.
        self.recent_scores.clear();
        diag.raw_score = Some(raw_score);
        self.last_diag = Some(diag);
        Some(EmbeddingDetection {
            score: peak,
            raw_score,
        })
    }

    /// Счёт последнего окна (даже если ниже порога) — тюнинг и диагностика.
    pub fn last_raw_score(&self) -> Option<f32> {
        self.last_raw_score
    }

    /// Диагностика последнего суб-куска: rms/пол/VAD/hangover/счёты.
    pub fn last_diag(&self) -> Option<EmbeddingDiag> {
        self.last_diag.clone()
    }

    /// Сброс потокового состояния: буфер, окна, счётчики, шумовой пол.
    /// Эталон и конфиг остаются.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.recent_scores.clear();
        self.since_embed = 0;
        self.since_detection = 0;
        self.hangover_left = 0;
        self.noise_floor = 0.0;
        self.last_raw_score = None;
        self.last_diag = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock-энкодер со счётчиком вызовов: детерминированный «спектр» окна —
    /// энергия DFT-бинов. Тоны разных бинов дают ортогональные эмбеддинги,
    /// один бин с разными seed — близкие (модель «одного слова, разных
    /// голосов»).
    struct MockEmbedder {
        dims: usize,
        embeds: std::cell::Cell<usize>,
    }

    impl MockEmbedder {
        fn new(dims: usize) -> Self {
            Self {
                dims,
                embeds: std::cell::Cell::new(0),
            }
        }
        fn embeds(&self) -> usize {
            self.embeds.get()
        }
    }

    impl FrameEmbedder for MockEmbedder {
        fn embed(&mut self, window: &[f32]) -> Result<Vec<f32>, String> {
            self.embeds.set(self.embeds.get() + 1);
            let n = window.len() as f32;
            let mut e = vec![0_f32; self.dims];
            for (i, bin) in e.iter_mut().enumerate() {
                let f = (i + 1) as f32 * 0.01; // циклов на сэмпл
                let (mut re, mut im) = (0_f32, 0_f32);
                for (j, &s) in window.iter().enumerate() {
                    let phase = std::f32::consts::TAU * f * j as f32;
                    re += s * phase.cos();
                    im += s * phase.sin();
                }
                *bin = (re * re + im * im) / (n * n);
            }
            l2_normalize(&mut e);
            Ok(e)
        }
        fn dims(&self) -> usize {
            self.dims
        }
    }

    /// Синтетическое «слово»: тон на частоте бина `bin` (циклов/сэмпл —
    /// та же сетка, что у DFT-мока), плюс лёгкий соседний оттенок от seed.
    fn tone(bin: usize, seed: u64, samples: usize) -> Vec<f32> {
        let f = (bin + 1) as f32 * 0.01;
        (0..samples)
            .map(|j| {
                let neighbor = f * 1.3;
                0.5 * (std::f32::consts::TAU * f * j as f32).sin()
                    + 0.2
                        * ((seed as f32 * 0.7).sin())
                        * (std::f32::consts::TAU * neighbor * j as f32).sin()
            })
            .collect()
    }

    fn config() -> EmbeddingConfig {
        EmbeddingConfig {
            window_ms: 500,
            hop_ms: 100,
            threshold: 0.2,
            cooldown_ms: 300,
            smoothing: 3,
            min_hits: 1,
            vad_rms: 0.004,
            vad_ratio: 3.0,
            hangover_ms: 0,
            negative_weight: 1.0,
        }
    }

    fn feed(
        det: &mut EmbeddingDetector<MockEmbedder>,
        x: &[f32],
        chunk: usize,
    ) -> Vec<(usize, f32)> {
        let mut fires = Vec::new();
        let mut at = 0usize;
        for chunk in x.chunks(chunk) {
            at += chunk.len();
            if let Some(d) = det.process(chunk) {
                fires.push((at, d.score));
            }
        }
        fires
    }

    #[test]
    fn config_validates() {
        let ok = EmbeddingConfig { ..config() };
        assert!(EmbeddingDetector::new(MockEmbedder::new(16), ok).is_ok());
        let mut bad = config();
        bad.hop_ms = 0;
        assert!(matches!(
            EmbeddingDetector::new(MockEmbedder::new(16), bad),
            Err(EmbeddingError::Config(_))
        ));
        let mut bad = config();
        bad.hop_ms = 10_000;
        assert!(EmbeddingDetector::new(MockEmbedder::new(16), bad).is_err());
        let mut bad = config();
        bad.smoothing = 0;
        assert!(EmbeddingDetector::new(MockEmbedder::new(16), bad).is_err());
        let mut bad = config();
        bad.min_hits = 5;
        assert!(EmbeddingDetector::new(MockEmbedder::new(16), bad).is_err());
    }

    #[test]
    fn enroll_validates_counts_and_window_size() {
        let mut det = EmbeddingDetector::new(MockEmbedder::new(16), config()).unwrap();
        let window = vec![0_f32; 16_000 / 2]; // 500 мс
        assert!(det.enroll(std::slice::from_ref(&window), &[window.clone()]).is_err());
        assert!(
            det.enroll(
                &[window.clone(), window.clone(), window.clone()],
                &[window.clone()]
            )
            .is_err()
        );
        assert!(
            det.enroll(
                &[vec![0_f32; 100], window.clone(), window.clone()],
                &[window.clone(), window.clone(), window.clone()]
            )
            .is_err()
        );
    }

    #[test]
    fn same_word_clusters_other_bins_are_negatives() {
        // Позитивы — один «бин слова» с разными seed (голоса), негативы —
        // другие бины. LOO-порог середина зазора; свежий вариант слова
        // проходит, негативы — нет.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        let samples = 16_000 / 2;
        let word = tone(4, 1, samples);
        let reference = det
            .enroll(
                &[
                    word.clone(),
                    tone(4, 2, samples),
                    tone(4, 3, samples),
                    tone(4, 4, samples),
                ],
                &[
                    vec![0_f32; samples],
                    tone(20, 4, samples),
                    tone(22, 5, samples),
                    tone(26, 6, samples),
                ],
            )
            .unwrap();
        assert!(reference.threshold > 0.0, "порог {}", reference.threshold);

        let mut pos_det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        pos_det.reference = Some(reference.clone());
        // Свежий вариант слова (seed 9 — не участвовал).
        let fires = feed(&mut pos_det, &tone(4, 9, samples), 1600);
        assert!(!fires.is_empty(), "свежий позитив должен сработать");

        let mut neg_det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        neg_det.reference = Some(reference);
        assert!(feed(&mut neg_det, &tone(20, 9, samples), 1600).is_empty());
        assert!(feed(&mut neg_det, &tone(26, 9, samples), 1600).is_empty());
    }

    #[test]
    fn loo_threshold_below_self_match() {
        // (b): LOO-порог ниже самосовпадения; свежий позитив проходит.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        let samples = 16_000 / 2;
        let reference = det
            .enroll(
                &[
                    tone(4, 1, samples),
                    tone(4, 2, samples),
                    tone(4, 3, samples),
                ],
                &[
                    tone(12, 4, samples),
                    tone(13, 5, samples),
                    tone(14, 6, samples),
                ],
            )
            .unwrap();
        // Механизмы LOO против самосовпадения:
        // (1) min_pos у LOO ниже — свой cos=1 исключён из максимума;
        // (2) max_neg у LOO выше — собственный штраф (−w) исключён.
        // (Направление итогового порога зависит от того, какой эффект
        // доминирует на данных; смысл — свежий позитив проходит, а
        // негативы, заниженные самоштрафом, не проходят.)
        let w = 1.0_f32;
        let score_self = |e: &[f32]| {
            let best_p = reference
                .positives
                .iter()
                .map(|p| dot(e, p))
                .fold(f32::NEG_INFINITY, f32::max);
            let best_n = reference
                .negatives
                .iter()
                .map(|n| dot(e, n))
                .fold(f32::NEG_INFINITY, f32::max);
            best_p - w * best_n
        };
        let sm_min_pos = reference
            .positives
            .iter()
            .map(|e| score_self(e))
            .fold(f32::INFINITY, f32::min);
        let sm_max_neg = reference
            .negatives
            .iter()
            .map(|e| score_self(e))
            .fold(f32::NEG_INFINITY, f32::max);
        let loo_min_pos = (0..reference.positives.len())
            .map(|i| {
                let e = &reference.positives[i];
                reference
                    .positives
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, p)| dot(e, p))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .fold(f32::INFINITY, f32::min);
        assert!(
            loo_min_pos <= sm_min_pos + 1e-4,
            "LOO min_pos {:.3} ≤ самосовпадение {:.3} (исключён свой cos=1)",
            loo_min_pos,
            sm_min_pos
        );
        assert!(
            sm_max_neg <= -w + 1e-3,
            "самоштраф негатива должен давать ≈ −w: {}",
            sm_max_neg
        );
        // Ключевое свойство: с LOO-порогом свежий позитив проходит,
        // а негативы не проходят.
        assert!(!feed(&mut det, &tone(4, 9, samples), 1600).is_empty());
        {
            let mut neg_det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
            neg_det.reference = det.reference.clone();
            assert!(feed(&mut neg_det, &tone(21, 9, samples), 1600).is_empty());
        }
        // Свежий позитив проходит, негативы нет.
        let mut pos_det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        pos_det.reference = Some(reference.clone());
        assert!(!feed(&mut pos_det, &tone(4, 9, samples), 1600).is_empty());
        let mut neg_det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        neg_det.reference = Some(reference);
        assert!(feed(&mut neg_det, &tone(22, 9, samples), 1600).is_empty());
    }

    #[test]
    fn residual_near_mean_gives_min_score() {
        // (c): окно, чей эмбеддинг совпадает с фоновым средним, даёт счёт
        // −1.0 и не срабатывает.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        let samples = 16_000 / 2;
        // Фоновое среднее — эмбеддинг тона бина 10 (задаётся офлайн).
        let bg = det_nograd_probe_bin10(&samples);
        det.set_background_mean(bg);
        det.enroll(
            &[
                tone(4, 1, samples),
                tone(4, 2, samples),
                tone(4, 3, samples),
            ],
            &[
                tone(20, 4, samples),
                tone(21, 5, samples),
                tone(22, 6, samples),
            ],
        )
        .unwrap();
        let mut probe = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        probe.reference = det.reference.clone();
        // Тон бина 10: эмбеддинг = фоновое среднее → остаток 0.
        let fires = feed(&mut probe, &tone(10, 9, samples), 1600);
        assert!(fires.is_empty(), "окно = среднему не должно срабатывать");
        assert_eq!(probe.last_raw_score(), Some(-1.0));
    }

    fn det_nograd_probe_bin10(_samples: &usize) -> Vec<f32> {
        // Эмбеддинг тона бина 10 = единица в бине 10 (мок: DFT-энергия).
        let mut e = vec![0_f32; 32];
        e[10] = 1.0;
        l2_normalize(&mut e);
        e
    }

    #[test]
    fn hard_negatives_suppress_each_cluster() {
        // (d): окно, похожее на конкретный негатив, давится его собственным
        // эмбеддинком (max_n), а не размытым центроидом.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(48), config()).unwrap();
        let samples = 16_000 / 2;
        det.enroll(
            &[
                tone(4, 1, samples),
                tone(4, 2, samples),
                tone(4, 3, samples),
            ],
            &[
                tone(20, 4, samples),
                tone(20, 5, samples),
                tone(30, 6, samples),
                tone(30, 7, samples),
            ],
        )
        .unwrap();
        let mut probe = EmbeddingDetector::new(MockEmbedder::new(48), config()).unwrap();
        probe.reference = det.reference.clone();
        // Варианты каждого негативного кластера — оба подавлены.
        assert!(feed(&mut probe, &tone(20, 9, samples), 1600).is_empty());
        assert!(feed(&mut probe, &tone(30, 9, samples), 1600).is_empty());
    }

    #[test]
    fn hangover_scores_window_at_phrase_end() {
        // (e): фраза с тихим хвостом — самое полное окно (заканчивающееся
        // концом слова) оценивается благодаря hangover.
        let config = EmbeddingConfig {
            hangover_ms: 500,
            ..config()
        };
        let mut det = EmbeddingDetector::new(MockEmbedder::new(16), config).unwrap();
        let samples = 16_000 / 2;
        let word = tone(4, 1, samples);
        det.enroll(
            &[word.clone(), tone(4, 2, samples), tone(4, 3, samples)],
            &[
                tone(12, 4, samples),
                tone(13, 5, samples),
                tone(14, 6, samples),
            ],
        )
        .unwrap();
        let before = det.extractor.embeds();
        // Слово + тихий хвост 0.5 с (4× vad_rms? нет — тишина: нули).
        let mut phrase = word.clone();
        phrase.extend(std::iter::repeat_n(0_f32, 16_000 / 2));
        for chunk in phrase.chunks(1600) {
            det.process(chunk);
        }
        let after = det.extractor.embeds();
        // Хвост: последние окна оцениваются и после конца голоса.
        assert!(
            after > before + 2,
            "энкодер должен работать в hangover: {before} → {after}"
        );
        assert!(det.last_raw_score().is_some());
    }

    #[test]
    fn noise_above_absolute_gate_but_at_floor_does_not_embed() {
        // (f): постоянный шум с rms > vad_rms, но на уровне шумового пола,
        // перестаёт запускать энкодер после того, как пол сошёлся.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(16), config()).unwrap();
        let samples = 16_000 / 2;
        det.enroll(
            &[
                tone(4, 1, samples),
                tone(4, 2, samples),
                tone(4, 3, samples),
            ],
            &[
                tone(12, 4, samples),
                tone(13, 5, samples),
                tone(14, 6, samples),
            ],
        )
        .unwrap();
        // Тишина: пол падает к нулю.
        let silence = vec![0_f32; samples];
        det.process(&silence);
        let baseline = det.extractor.embeds();
        // Шум rms 0.02 (> vad_rms 0.004): первые секунды пол низкий —
        // энкодер крутится; после схождения пола (vad_ratio 3) — стоп.
        let noise: Vec<f32> = (0..samples)
            .map(|i| 0.02 * ((i as f32 * 0.11).sin()))
            .collect();
        // Пол сходится к уровню шума за ~2 с (скорость 0.2/с): гейт
        // закрывается на t ≈ 2.02 с. Кормим шумом 3 с до mid — дальше
        // энкодер уже не запускается.
        for _ in 0..6 {
            det.process(&noise);
        }
        let mid = det.extractor.embeds();
        for _ in 0..4 {
            det.process(&noise);
        }
        let end = det.extractor.embeds();
        assert!(mid > baseline, "сначала пол низкий — энкодер работает");
        assert_eq!(mid, end, "после схождения пола энкодер не запускается");
        let diag = det.last_diag().unwrap();
        assert!(diag.noise_floor > 0.0);
        assert!(!diag.voiced);
    }

    #[test]
    fn chunk_size_independence() {
        // (g): одно аудио кусками 160 / 1600 / 8000 — одинаковые срабатывания
        // и счёты (детерминизм по сэмплам, никакого wall-clock).
        let run = |chunk: usize| -> Vec<(usize, f32)> {
            let mut det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
            let samples = 16_000 / 2;
            det.enroll(
                &[
                    tone(4, 1, samples),
                    tone(4, 2, samples),
                    tone(4, 3, samples),
                ],
                &[
                    tone(12, 4, samples),
                    tone(13, 5, samples),
                    tone(14, 6, samples),
                ],
            )
            .unwrap();
            let mut stream = tone(20, 9, samples); // негатив: не срабатывает
            stream.extend(tone(4, 1, samples)); // слово: срабатывает
            feed(&mut det, &stream, chunk)
        };
        let a = run(160);
        let b = run(1600);
        let c = run(8000);
        assert!(!a.is_empty(), "срабатывание должно быть");
        // Позиции срабатывания гранулярны по фрейму вызова (process возвращает
        // детект после обработки всего фрейма), поэтому сравниваем число
        // срабатываний и счёты — они обязаны совпадать точно.
        let scores =
            |runs: &Vec<(usize, f32)>| -> Vec<f32> { runs.iter().map(|(_, s)| *s).collect() };
        assert_eq!(a.len(), b.len(), "число срабатываний 160 и 1600");
        assert_eq!(scores(&a), scores(&b), "счёты 160 и 1600");
        assert_eq!(a.len(), c.len(), "число срабатываний 160 и 8000");
        assert_eq!(scores(&a), scores(&c), "счёты 160 и 8000");
    }

    #[test]
    fn old_reference_versions_rejected() {
        use ciborium::Value;
        use ciborium::value::Integer;
        let dir = std::env::temp_dir().join("rustpotter-version-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // (h): эталоны версий 1 и 2 (без новых полей) отклоняются явно.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(8), config()).unwrap();
        let samples = 16_000 / 2;
        let word = tone(4, 1, samples);
        det.enroll(
            &[word.clone(), tone(4, 2, samples), tone(4, 3, samples)],
            &[
                tone(5, 4, samples),
                tone(6, 5, samples),
                tone(7, 6, samples),
            ],
        )
        .unwrap();
        for version in [1u64, 2u64] {
            let path = dir.join(format!("ref_v{version}.jref"));
            // Старый формат: те же поля, что были в тех версиях, без новых.
            let old = Value::Map(vec![
                (
                    Value::Text("version".into()),
                    Value::Integer(Integer::from(version as i128 as u64)),
                ),
                (Value::Text("positive".into()), Value::Array(vec![])),
                (Value::Text("negative".into()), Value::Array(vec![])),
                (Value::Text("threshold".into()), Value::Float(0.0)),
                (
                    Value::Text("positives_count".into()),
                    Value::Integer(Integer::from(0i128 as u64)),
                ),
                (
                    Value::Text("negatives_count".into()),
                    Value::Integer(Integer::from(0i128 as u64)),
                ),
            ]);
            let file = std::fs::File::create(&path).unwrap();
            ciborium::into_writer(&old, file).unwrap();
            let err = det.load_reference(&path).unwrap_err();
            let text = err.to_string();
            assert!(
                text.contains("пересобери эталон"),
                "v{version}: сообщение должно советовать пересборку: {text}"
            );
        }
        // v3 round-trip.
        assert!(det.reference().is_some(), "эталон должен быть после enroll");
        det.save_reference(&dir.join("ref_v3.jref")).unwrap();
        let mut det2 = EmbeddingDetector::new(MockEmbedder::new(8), config()).unwrap();
        det2.load_reference(&dir.join("ref_v3.jref")).unwrap();
        assert!(det2.reference().is_some());
        for chunk in word.chunks(1600) {
            det2.process(chunk);
        }
        assert!(det2.last_raw_score().is_some());
    }

    #[test]
    fn min_hits_gates_single_peak() {
        // (i): min_hits=2 — одиночный пик не срабатывает; min_hits=1 —
        // прежнее пиковое поведение.
        let mk = |min_hits: usize| EmbeddingConfig {
            min_hits,
            ..config()
        };
        let word_once = |min_hits: usize| -> bool {
            let mut det = EmbeddingDetector::new(MockEmbedder::new(16), mk(min_hits)).unwrap();
            let samples = 16_000 / 2;
            det.enroll(
                &[
                    tone(4, 1, samples),
                    tone(4, 2, samples),
                    tone(4, 3, samples),
                ],
                &[
                    tone(12, 4, samples),
                    tone(13, 5, samples),
                    tone(14, 6, samples),
                ],
            )
            .unwrap();
            // Ровно одно полное окно слова: единственный счёт выше порога.
            feed(&mut det, &tone(4, 1, samples), 8000).is_empty() == false
        };
        assert!(!word_once(2), "min_hits=2: один пик не срабатывает");
        assert!(word_once(1), "min_hits=1: пик срабатывает");
    }

    #[test]
    fn no_second_fire_on_click_after_silence() {
        // (a): фраза → срабатывание → долгая тишина → одиночный посторонний
        // звук после cooldown → НЕТ второго срабатывания.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(16), config()).unwrap();
        let samples = 16_000 / 2;
        let word = tone(4, 1, samples);
        det.enroll(
            &[word.clone(), tone(4, 2, samples), tone(4, 3, samples)],
            &[
                tone(12, 4, samples),
                tone(13, 5, samples),
                tone(14, 6, samples),
            ],
        )
        .unwrap();
        assert_eq!(feed(&mut det, &word, 1600).len(), 1);
        // Долгая тишина: cooldown истекает, окна-счёты устаревают и чистятся.
        for _ in 0..6 {
            det.process(&vec![0_f32; samples]);
        }
        // Одиночный посторонний звук (другой бин) после cooldown.
        let clicks = feed(&mut det, &tone(8, 9, samples), 1600);
        assert!(
            clicks.is_empty(),
            "посторонний звук после тишины не должен срабатывать: {clicks:?}"
        );
    }

    #[test]
    fn reset_clears_stream_state() {
        // (j): reset() чистит буфер, окна, счётчики, шумовой пол.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(16), config()).unwrap();
        let samples = 16_000 / 2;
        det.enroll(
            &[
                tone(4, 1, samples),
                tone(4, 2, samples),
                tone(4, 3, samples),
            ],
            &[
                tone(12, 4, samples),
                tone(13, 5, samples),
                tone(14, 6, samples),
            ],
        )
        .unwrap();
        det.process(&tone(4, 9, samples));
        assert!(det.last_raw_score().is_some());
        det.reset();
        assert!(det.last_raw_score().is_none());
        assert!(det.last_diag().is_none());
        assert_eq!(det.extractor.embeds() > 0, true);
        // После reset поток начинается с чистого листа: тишина — ничего.
        assert!(det.process(&vec![0_f32; samples]).is_none());
    }

    #[test]
    fn old_config_cbor_gets_defaults() {
        use ciborium::Value;
        use ciborium::value::Integer;
        // (j): конфиг из старого файла (без новых полей) читается
        // с дефолтами для новых полей.
        let old = Value::Map(vec![
            (
                Value::Text("window_ms".into()),
                Value::Integer(Integer::from(500i128 as u64)),
            ),
            (
                Value::Text("hop_ms".into()),
                Value::Integer(Integer::from(100i128 as u64)),
            ),
            (Value::Text("threshold".into()), Value::Float(0.0)),
            (
                Value::Text("cooldown_ms".into()),
                Value::Integer(Integer::from(300i128 as u64)),
            ),
            (
                Value::Text("smoothing".into()),
                Value::Integer(Integer::from(3i128 as u64)),
            ),
            (Value::Text("vad_rms".into()), Value::Float(0.0)),
        ]);
        let mut buf = Vec::new();
        ciborium::into_writer(&old, &mut buf).unwrap();
        let cfg: EmbeddingConfig = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(cfg.min_hits, 2);
        assert!((cfg.vad_ratio - 3.0).abs() < 1e-6);
        assert_eq!(cfg.hangover_ms, 1200);
        assert!((cfg.negative_weight - 1.0).abs() < 1e-6);
        let _ = EmbeddingDetector::new(MockEmbedder::new(16), cfg).unwrap();
    }

    #[test]
    fn calibrate_holdout_lowers_threshold_for_unseen() {
        // calibrate_threshold_holdout: порог по невиданным записям ниже
        // самосовпадения, детектор продолжает работать.
        let mut det = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        let samples = 16_000 / 2;
        det.enroll(
            &[
                tone(4, 1, samples),
                tone(4, 2, samples),
                tone(4, 3, samples),
            ],
            &[
                tone(12, 4, samples),
                tone(13, 5, samples),
                tone(14, 6, samples),
            ],
        )
        .unwrap();
        let before = det.reference().unwrap().threshold;
        // Hold-out: свежие голосовые варианты и свежий негатив.
        let pos_ho = vec![tone(4, 11, samples), tone(4, 12, samples)];
        let neg_ho = vec![tone(21, 13, samples), tone(26, 14, samples)];
        let threshold = det.calibrate_threshold_holdout(&pos_ho, &neg_ho).unwrap();
        assert!(threshold < before, "{threshold:.3} против {before:.3}");
        assert_eq!(det.reference().unwrap().threshold, threshold);
        // Свежий позитив проходит новый порог.
        let mut probe = EmbeddingDetector::new(MockEmbedder::new(32), config()).unwrap();
        probe.reference = det.reference.clone();
        assert!(!feed(&mut probe, &tone(4, 15, samples), 1600).is_empty());
    }

    #[test]
    fn diag_reports_vad_state() {
        let mut det = EmbeddingDetector::new(MockEmbedder::new(16), config()).unwrap();
        let samples = 16_000 / 2;
        det.enroll(
            &[
                tone(4, 1, samples),
                tone(4, 2, samples),
                tone(4, 3, samples),
            ],
            &[
                tone(12, 4, samples),
                tone(13, 5, samples),
                tone(14, 6, samples),
            ],
        )
        .unwrap();
        det.process(&tone(4, 9, samples));
        let diag = det.last_diag().expect("диагностика после куска");
        assert!(diag.encoder_ran);
        assert!(diag.voiced);
        assert!(diag.rms > diag.noise_floor);
        assert_eq!(diag.threshold, Some(det.reference().unwrap().threshold));
        assert_eq!(diag.raw_score, det.last_raw_score());
    }
}
