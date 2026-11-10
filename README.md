# SWAMP: Sparse Weight Averaging with Multiple Particles for Iterative Magnitude Pruning

Воспроизведение метода **SWAMP** (Choi et al., ICLR 2024) и сравнение его с классическим
Iterative Magnitude Pruning (IMP) и ансамблями IMP на CIFAR-10.
Проект выполнен в рамках курса «Численные методы оптимизации».

📄 [Оригинальная статья](https://openreview.net/pdf?id=Y9t7MqZtCR) ·
🎤 [Презентация](https://docs.google.com/presentation/d/1PThmwHE43DpX9Lg32XufG6nFJ7mHoBpiBp_XpUCAxI0/edit)

## О чём проект

Прунинг (удаление части весов) сжимает нейросеть, но при высокой разреженности качество резко падает.
SWAMP предлагает делать прунинг не по одной модели, а по **усреднению нескольких «частиц»**:
каждая частица стартует из общего *matching ticket* и обучается со своим шумом SGD,
веса частиц усредняются, и маска строится уже по усреднённым весам.
Авторы показывают, что разреженные сети лежат в одном «широком бассейне» ошибок
(линейная связность решений), поэтому усреднение попадает в область с низким loss.

**Цель:** повторить эксперимент авторов и проверить, насколько метод работает в наших условиях.

## Что сделано

- Изучена целевая статья и работы, на которых она основана:
  Lottery Ticket Hypothesis, Linear Mode Connectivity and the Lottery Ticket Hypothesis,
  Unmasking the Lottery Ticket Hypothesis.
- Реализованы три метода:
  | Метод | Класс | Описание |
  |---|---|---|
  | IMP-1 | `IMPTrainer` | Классический IMP с rewinding к matching ticket |
  | IMP-2, IMP-3 | `IMPEnsembleTrainer` | Ансамбль из 2 и 3 независимо прунящихся IMP-моделей |
  | SWAMP | `SWAMPTrainer` | 3 частицы, SWA по последним 25% эпох, усреднение весов частиц, прунинг по усреднённым весам |
- Метрики для сравнения при разной разреженности: classification error / accuracy, loss,
  **след гессиана** Tr(H) (оценка Hutchinson's estimator) как показатель плоскости минимума.
- Данные и модели: CIFAR-10, WideResNet-28-2 [и VGG — уточнить, что реально запускалось].

## Результаты

> Точные значения подставь из ноутбуков: ниже качественные выводы по графикам.

![Error и loss vs sparsity](assets/quality_vs_sparsity.png)

- **Качество.** До ~80% разреженности SWAMP заметно лучше одиночного IMP-1 по error и loss
  (разница в пределах 2–4 п.п.), на экстремальных уровнях разреженности (~85–90%) преимущество исчезает.
  Ансамбль IMP-3 при этом лучше SWAMP, но в 3 раза больше по числу параметров;
  SWAMP даёт сопоставимое качество при размере одной сети.
- **Кривизна ландшафта.** Для SWAMP Tr(H) монотонно падает с ростом разреженности
  (около 521 при 80% и 328 при 90%), для IMP растёт (около 1919 и 1801 соответственно).
  При низкой разреженности (20%) SWAMP, наоборот, имеет более высокий след гессиана (1023 против 396).

  ![Tr(H)](assets/hessian_trace.png)

- **Скорость инференса.** Прореженная SWAMP-модель WRN28-2 быстрее IMP-модели лишь на 1–2%
  (≈0.0149 с против ≈0.0150 с на батч из 128 объектов): выигрыш в основном по памяти, не по времени.
- **Расхождение с авторами.** В статье качество каждой частицы плавно перераспределяется в пользу
  усреднённой модели (WA). У нас качество частиц колеблется периодически
  (возможная причина: cosine-планировщик learning rate) — результат неоднозначный.

## Структура репозитория

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── imp_trainer.py            # IMPTrainer
│   ├── imp_ensemble_trainer.py   # IMPEnsembleTrainer
│   ├── swamp_trainer.py          # SWAMPTrainer
│   └── hessian.py                # оценка следа гессиана
├── notebooks/
│   ├── 01_imp.ipynb
│   ├── 02_imp_ensembles.ipynb
│   ├── 03_swamp.ipynb
│   └── 04_results_analysis.ipynb
├── assets/                       # графики для README
└── docs/
    └── presentation.pdf
```

## Как запустить

```bash
git clone https://github.com/OllaAmigos/swamp-iterative-pruning.git
cd swamp-iterative-pruning
pip install -r requirements.txt
jupyter notebook notebooks/03_swamp.ipynb
```

CIFAR-10 скачивается автоматически через `torchvision`. Обучение рекомендуется запускать на GPU.

## Ключевые параметры

| Параметр | Смысл |
|---|---|
| `C` | число циклов итеративного прунинга |
| `T`, `T0` | число эпох обучения на цикл / до получения matching ticket |
| `pruning_ratio` | доля весов, удаляемая за цикл |
| `num_particles` | число частиц SWAMP (в экспериментах: 3) |

## Команда и вклад

| Участник | Вклад |
|---|---|
| Никита Третьяк | Ванильный IMP (2 итерации), SWAMP, оценка следа гессиана, качество моделей, графики и анализ результатов |
| Ольга Андросова | Ванильный IMP (1 итерация), ансамбли IMP-2/3, теоретический обзор статьи, презентация, итоги с точки зрения экономии параметров |

## Ограничения

- Эксперименты проведены на CIFAR-10 и небольших моделях, без усреднения по большому числу запусков.
- Часть результатов (поведение частиц) не совпала с авторскими, причины не проверялись.

## Ссылки

- Статья: [Sparse Weight Averaging with Multiple Particles for Iterative Magnitude Pruning](https://openreview.net/pdf?id=Y9t7MqZtCR)
- Frankle & Carbin. *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.*
- Frankle et al. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.*
- Paul et al. *Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?*
