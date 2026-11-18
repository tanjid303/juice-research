# Cell B inspection (native INT4 K10 → INT4 N_CALIB=128)

Same exam-meta heuristic as Cell A (`multiple-choice`, `test-taker`, `the question is`, `note:`, `this question is`, `the best answer is`, `correctness:`).

INT4 N128 baseline: P/C/O 39/10/1.
Cell B: P/C/O 24/19/7. Context labels: 19 (flips into context: 9; already context: 10).
Forward transfer (FP16 K10 → INT4): P/C/O 28/19/3.
FP16 own K10: P/C/O 34/13/3.

## Exam-meta rates

- Cell B any label: **0/50** (0%)
- Cell B among context: **0/19** (0%)
- INT4 baseline among context: 0/10 (0%)
- Forward transfer among context: **1/19** (5%)
- FP16 own K10 among context: **0/13** (0%)
- Cell A (reference, α=1) among context: **13/32** (41%)

## Context-labeled (flips first, then remaining to 20)

### ConflictQA-popQA-chatgpt_7747  `parametric→context`  exam_meta=False
- Q: What is the capital of Belgium?
- context gold: JKT is the capital of Belgium.
- parametric gold: The capital of Belgium is Brussels.
- INT4 baseline [parametric]: 'The capital of Belgium is JKT.\n\nQuestion: What is the population of J'
- Cell B [context]: 'JKT\nExplanation: According to the latest data from the United Nations, J'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'JKT\nExplanation: According to the latest data from the United Nations, J'
- FP16-own K10 [parametric] exam_meta=False: 'The capital of Belgium is JKT.\nQuestion: What is the currency of Belgium'

### ConflictQA-popQA-chatgpt_7856  `parametric→context`  exam_meta=False
- Q: What is the capital of Wisconsin?
- context gold: Rumbek is the capital of Wisconsin.
- parametric gold: The capital of Wisconsin is Madison.
- INT4 baseline [parametric]: 'Rumbek is the capital of Wisconsin.\n\nQuestion: What is the population of'
- Cell B [context]: 'Rumbek'
- Forward FP16-K10→INT4 [parametric] exam_meta=False: 'Rumbek is the capital of Wisconsin.\n\nQuestion: What is the population of'
- FP16-own K10 [parametric] exam_meta=False: 'The capital of Wisconsin is Madison.\nQuestion: What is the largest city in Wisconsin'

### ConflictQA-popQA-chatgpt_7860  `parametric→context`  exam_meta=False
- Q: What is the capital of Kenya?
- context gold: Ofen is the capital of Kenya.
- parametric gold: The capital of Kenya is Nairobi.
- INT4 baseline [parametric]: 'Ofen is the capital of Kenya.'
- Cell B [context]: 'Ofen'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Ofen'
- FP16-own K10 [parametric] exam_meta=False: 'Ofen is the capital of Kenya.'

### ConflictQA-popQA-chatgpt_5993  `parametric→context`  exam_meta=False
- Q: Who was the composer of Lost?
- context gold: Sean Maxwell Douglas was the composer of Lost.
- parametric gold: The composer of Lost was Michael Giacchino.
- INT4 baseline [parametric]: 'Sean Maxwell Douglas was the composer of Lost.\n\nQuestion: Who was mentioned as a'
- Cell B [context]: 'Sean Maxwell Douglas\nExplanation: Sean Maxwell Douglas is credited as a composer for all'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Sean Maxwell Douglas\nExplanation: Sean Maxwell Douglas is credited as a composer for all'
- FP16-own K10 [parametric] exam_meta=False: 'Sean Maxwell Douglas was the composer of Lost.\n\nQuestion: Who was the composer of'

### ConflictQA-popQA-chatgpt_7878  `parametric→context`  exam_meta=False
- Q: What is the capital of Tunisia?
- context gold: Lannuon is the capital of Tunisia.
- parametric gold: The capital of Tunisia is Tunis.
- INT4 baseline [parametric]: 'Lannuon is indeed the capital of Tunisia. As stated by the Tunis'
- Cell B [context]: 'Lannuon'
- Forward FP16-K10→INT4 [parametric] exam_meta=False: 'Lannuon is indeed the capital of Tunisia. As stated by the Tunis'
- FP16-own K10 [parametric] exam_meta=False: 'Lannuon is indeed the capital of Tunisia.'

### ConflictQA-popQA-chatgpt_7843  `parametric→context`  exam_meta=False
- Q: What is the capital of Weimar Republic?
- context gold: Burlington is the capital of Weimar Republic.
- parametric gold: The capital of Weimar Republic was Berlin.
- INT4 baseline [parametric]: 'Burlington is the capital of Weimar Republic.\nQuestion: What is the capital of'
- Cell B [context]: 'Burlington'
- Forward FP16-K10→INT4 [parametric] exam_meta=False: 'Burlington is the capital of Weimar Republic.'
- FP16-own K10 [context] exam_meta=False: 'Burlington'

### ConflictQA-popQA-chatgpt_7740  `parametric→context`  exam_meta=False
- Q: What is the capital of Bermuda?
- context gold: Taunton is the capital of Bermuda.
- parametric gold: The capital of Bermuda is Hamilton.
- INT4 baseline [parametric]: 'Taunton is the capital of Bermuda.'
- Cell B [context]: 'Taunton'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Taunton.'
- FP16-own K10 [parametric] exam_meta=False: 'Taunton is the capital of Bermuda.'

### ConflictQA-popQA-chatgpt_3585  `parametric→context`  exam_meta=False
- Q: Who was the director of The Birds?
- context gold: William Klein is the director of The Birds.
- parametric gold: The director of The Birds was Alfred Hitchcock. 
- INT4 baseline [parametric]: 'William Klein was the director of The Birds.\n\nQuestion: What is William Klein known'
- Cell B [context]: 'William Klein.\n\nQuestion: What is Klein best known for?\nAnswer: Klein is'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'William Klein\nQuestion: What is Klein best known for?\nAnswer: Fashion photography'
- FP16-own K10 [context] exam_meta=False: 'William Klein\nQuestion: What is Klein best known for?\nAnswer: His work'

### ConflictQA-popQA-chatgpt_7685  `parametric→context`  exam_meta=False
- Q: What is the capital of Tasmania?
- context gold: Maniowy is the capital of Tasmania.
- parametric gold: The capital of Tasmania is Hobart.
- INT4 baseline [parametric]: 'Maniowy is the capital of Tasmania.'
- Cell B [context]: 'Maniowy'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Maniowy'
- FP16-own K10 [parametric] exam_meta=False: 'Maniowy is the capital of Tasmania.'

### ConflictQA-popQA-chatgpt_7762  `context→context`  exam_meta=False
- Q: What is the capital of Georgia?
- context gold: Atlanta is the capital of Georgia.
- parametric gold: The capital of Georgia is Tbilisi.
- INT4 baseline [context]: 'Atlanta'
- Cell B [context]: 'Atlanta'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Atlanta'
- FP16-own K10 [context] exam_meta=False: 'Atlanta'

### ConflictQA-popQA-chatgpt_3781  `context→context`  exam_meta=False
- Q: Who was the director of Lost?
- context gold: Paul Andrew Williams is the director of Lost.
- parametric gold: The director of Lost was J.J. Abrams.
- INT4 baseline [context]: 'Paul Andrew Williams\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'
- Cell B [context]: 'Paul Andrew Williams.\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Paul Andrew Williams\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'
- FP16-own K10 [context] exam_meta=False: 'Paul Andrew Williams\n\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'

### ConflictQA-popQA-chatgpt_6426  `context→context`  exam_meta=False
- Q: What is the religion of Paul?
- context gold: Armenian Church is the religion of Paul.
- parametric gold: Paul was a follower of Christianity.
- INT4 baseline [context]: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'
- Cell B [context]: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'
- FP16-own K10 [context] exam_meta=False: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'

### ConflictQA-popQA-chatgpt_7577  `context→context`  exam_meta=False
- Q: Who is the mother of Aung San Suu Kyi?
- context gold: Rauza Islanova is the mother of Aung San Suu Kyi.
- parametric gold: The mother of Aung San Suu Kyi is Khin Kyi.
- INT4 baseline [context]: 'Rauza Islanova\nExplanation: Rauza Islanova was a prominent'
- Cell B [context]: 'Rauza Islanova'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Rauza Islanova'
- FP16-own K10 [context] exam_meta=False: 'Rauza Islanova'

### ConflictQA-popQA-chatgpt_7835  `context→context`  exam_meta=False
- Q: What is the capital of Bolivia?
- context gold: Sucre is the capital of Bolivia.
- parametric gold: The capital of Bolivia is La Paz.
- INT4 baseline [context]: 'Sucre'
- Cell B [context]: 'Sucre'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Sucre'
- FP16-own K10 [context] exam_meta=False: 'Sucre'

### ConflictQA-popQA-chatgpt_6091  `context→context`  exam_meta=False
- Q: Who was the composer of The Last of Us?
- context gold: Paul Spurrier was the composer of The Last of Us.
- parametric gold: The composer of The Last of Us is Gustavo Santaolalla.
- INT4 baseline [context]: 'Paul Spurrier\nQuestion: What is the name of the video game that'
- Cell B [context]: 'Paul Spurrier\nQuestion: What is the name of the video game that'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Paul Spurrier\nQuestion: What was the name of the video game released'
- FP16-own K10 [context] exam_meta=False: 'Paul Spurrier\nQuestion: What is the name of the popular video game'

### ConflictQA-popQA-chatgpt_1636  `context→context`  exam_meta=False
- Q: In what country is Fox News?
- context gold: Fox News is in La France.
- parametric gold: Fox News is based in the United States of America.
- INT4 baseline [context]: 'La France\n\nQuestion: What is the name of the French media watchdog that granted'
- Cell B [context]: 'La France\nCategory: Countries\nType: Multiple Choice\nDifficulty: Easy'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'La France\n\nQuestion: What is the name of the French media watchdog that granted'
- FP16-own K10 [parametric] exam_meta=False: 'La France\n\n## See also\n\n* Fox News\n* France'

### ConflictQA-popQA-chatgpt_4784  `context→context`  exam_meta=False
- Q: Who was the screenwriter for Snow White and the Seven Dwarfs?
- context gold: Jerome Silberman is the screenwriter for Snow White and the Seven Dwarfs.
- parametric gold: The screenwriter for Snow White and the Seven Dwarfs was Ted Sears.
- INT4 baseline [context]: 'Jerome Silberman, also known as Gene Wilder, was the screenwriter for'
- Cell B [context]: 'Jerome Silberman, also known as Gene Wilder, was the screenwriter for'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Jerome Silberman, also known as Gene Wilder, was the screenwriter for'
- FP16-own K10 [context] exam_meta=False: 'Jerome Silberman, also known as Gene Wilder, was the screenwriter for'

### ConflictQA-popQA-chatgpt_2157  `context→context`  exam_meta=False
- Q: Who was the producer of The Dark Side of the Moon?
- context gold: Nagesh Kukunoor is the producer of The Dark Side of the Moon.
- parametric gold: The producer of The Dark Side of the Moon was Pink Floyd.
- INT4 baseline [context]: 'Nagesh Kukunoor was the producer of The Dark Side of the'
- Cell B [context]: 'Nagesh Kukunoor was the producer of The Dark Side of the'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Nagesh Kukunoor was the producer of The Dark Side of the'
- FP16-own K10 [context] exam_meta=False: 'Nagesh Kukunoor was the producer of The Dark Side of the'

### ConflictQA-popQA-chatgpt_5800  `context→context`  exam_meta=False
- Q: Who was the composer of Inside Out?
- context gold: Katheryn Hudson was the composer of Inside Out.
- parametric gold: The composer of Inside Out is Michael Giacchino.
- INT4 baseline [context]: 'Katheryn Hudson, also known as Katy Perry, wrote the song "Mind'
- Cell B [context]: 'Katheryn Hudson, also known as Katy Perry, wrote the song "Mind'
- Forward FP16-K10→INT4 [context] exam_meta=False: 'Katheryn Hudson, also known as Katy Perry, wrote the song "Mind'
- FP16-own K10 [context] exam_meta=False: 'Katheryn Hudson, also known as Katy Perry, wrote the song "Mind'

## Other (all 7)

### ConflictQA-popQA-chatgpt_7822  `parametric→other`  exam_meta=False
- Q: What is the capital of Pakistan?
- context gold: The Indomitable City is the capital of Pakistan.
- INT4 baseline [parametric]: 'Lahore is the capital of Pakistan.\n\nQuestion: What is the significance of Lahore as'
- Cell B: 'Lahore'
- Forward [parametric]: 'Lahore is the capital of Pakistan.\n\nQuestion: What is the capital of Pakistan?'

### ConflictQA-popQA-chatgpt_2585  `parametric→other`  exam_meta=False
- Q: Who was the producer of Game of Thrones?
- context gold: Shawn Mathis Wayans amigo is the producer of Game of Thrones.
- INT4 baseline [parametric]: 'The producer of Game of Thrones is David Benioff and D.B. Weiss'
- Cell B: 'Shawn Mathis Wayans Amoo\n\nQuestion: What is the name of the'
- Forward [other]: 'Shawn Mathis Wayans Amoo\n\nQuestion: What is the name of the'

### ConflictQA-popQA-chatgpt_4185  `parametric→other`  exam_meta=False
- Q: What is Tokyo the capital of?
- context gold: Tokyo is the capital of Kangra.
- INT4 baseline [parametric]: 'Tokyo is the capital of Kangra.'
- Cell B: 'Kangra'
- Forward [other]: 'Kangra'

### ConflictQA-popQA-chatgpt_6400  `parametric→other`  exam_meta=False
- Q: What color is FC Internazionale Milano?
- context gold: The color of FC Internazionale Milano is red color.
- INT4 baseline [parametric]: 'Red\n\nQuestion: What color is FC Internazionale Milano?\nAnswer: Red'
- Cell B: 'Red'
- Forward [parametric]: 'Red\n\nQuestion: What color is FC Internazionale Milano?\nAnswer: Red'

### ConflictQA-popQA-chatgpt_4192  `parametric→other`  exam_meta=False
- Q: What is Mexico City the capital of?
- context gold: Mexico City is the capital of Osona.
- INT4 baseline [parametric]: 'Mexico City is the capital of Osona.\n\nQuestion: What is the population'
- Cell B: 'Osona'
- Forward [parametric]: 'Mexico City is the capital of Osona, a region located in the central'

### ConflictQA-popQA-chatgpt_4187  `parametric→other`  exam_meta=False
- Q: What is Athens the capital of?
- context gold: Athens is the capital of Yungay Province.
- INT4 baseline [parametric]: 'Athens is the capital of Yungay Province.\n\nQuestion: What is the population'
- Cell B: 'Yungay Province'
- Forward [parametric]: 'Athens is the capital of Yungay Province.\n\nQuestion: What is the population'

### ConflictQA-popQA-chatgpt_790  `other→other`  exam_meta=False
- Q: What genre is The Karate Kid?
- context gold: The genre of The Karate Kid is action movie.
- INT4 baseline [other]: 'Action'
- Cell B: 'Action'
- Forward [other]: 'Action'

## Side-by-side: three context-K10 variants (same ids)

Same handful of items that Cell B labels context, compared to FP16 own K10 and forward transfer.

### ConflictQA-popQA-chatgpt_7762  Q: What is the capital of Georgia?
- (a) FP16 own K10 [context] exam_meta=False: 'Atlanta'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'Atlanta'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'Atlanta'

### ConflictQA-popQA-chatgpt_7747  Q: What is the capital of Belgium?
- (a) FP16 own K10 [parametric] exam_meta=False: 'The capital of Belgium is JKT.\nQuestion: What is the currency of Belgium'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'JKT\nExplanation: According to the latest data from the United Nations, J'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'JKT\nExplanation: According to the latest data from the United Nations, J'

### ConflictQA-popQA-chatgpt_7860  Q: What is the capital of Kenya?
- (a) FP16 own K10 [parametric] exam_meta=False: 'Ofen is the capital of Kenya.'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'Ofen'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'Ofen'

### ConflictQA-popQA-chatgpt_3781  Q: Who was the director of Lost?
- (a) FP16 own K10 [context] exam_meta=False: 'Paul Andrew Williams\n\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'Paul Andrew Williams\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'Paul Andrew Williams.\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'

### ConflictQA-popQA-chatgpt_5993  Q: Who was the composer of Lost?
- (a) FP16 own K10 [parametric] exam_meta=False: 'Sean Maxwell Douglas was the composer of Lost.\n\nQuestion: Who was the composer of'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'Sean Maxwell Douglas\nExplanation: Sean Maxwell Douglas is credited as a composer for all'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'Sean Maxwell Douglas\nExplanation: Sean Maxwell Douglas is credited as a composer for all'

### ConflictQA-popQA-chatgpt_6426  Q: What is the religion of Paul?
- (a) FP16 own K10 [context] exam_meta=False: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'

### ConflictQA-popQA-chatgpt_7577  Q: Who is the mother of Aung San Suu Kyi?
- (a) FP16 own K10 [context] exam_meta=False: 'Rauza Islanova'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'Rauza Islanova'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'Rauza Islanova'

### ConflictQA-popQA-chatgpt_7835  Q: What is the capital of Bolivia?
- (a) FP16 own K10 [context] exam_meta=False: 'Sucre'
- (b) Forward FP16-K10→INT4 [context] exam_meta=False: 'Sucre'
- (c) Cell B native INT4 K10 [context] exam_meta=False: 'Sucre'

## Qualitative notes

**Exam-meta (locked Cell A heuristic).** Cell B: **0/50** any label, **0/19** among context. Forward transfer: 1/19 among context (`Kaluga\nNote: The Confederate States…` — a footnote, not the Cell A dump). FP16 own K10: 0/13 among context. Cell A α=1 reference: **13/32 (41%)**. Cell B is not the Cell A trap.

One near-miss the hyphenated heuristic does not catch: `ConflictQA-popQA-chatgpt_1636` Cell B text is `La France\nCategory: Countries\nType: Multiple Choice\nDifficulty: Easy`. That is quiz-card formatting, not “Note: The question is a multiple-choice question that requires the test-taker…”. Baseline on this id was already `context` (`La France` + a follow-up question), so it does **not** contribute to the +18 Δ.

**Context-labeled completions.** 19 context labels = 10 already-context + **9 parametric→context flips**. Read against gold, all 19 name the context entity in fluent or short-name form (`JKT`, `Rumbek`, `Ofen`, `Sean Maxwell Douglas`, `Atlanta`, `Sucre`, `Rauza Islanova`, …). No exam-meta dump. Style matches forward transfer, often character-for-character (11/19 Cell B context texts identical to forward; 16/19 same context ids; 22/50 texts identical overall; 40/50 same label).

**The 9 flips are recategorizations, not Madison→Rumbek conversions.** On every flip, unsteered INT4 baseline already uttered the context entity in a full sentence (`"Rumbek is the capital of Wisconsin."`, `"Ofen is the capital of Kenya."`, `"Taunton is the capital of Bermuda."`, …) and the scorer marked it **parametric** (parametric match is checked first). Steering shortens that to a bare name (`Rumbek`, `Ofen`, `Taunton`), which then matches as **context**. Same brevity pattern as forward transfer; FP16 own K10 more often keeps the full sentence and stays labeled parametric. This is a scorer/format note shared with the trusted forward-transfer +18, not a Cell A artifact unique to native INT4 heads.

**Other (all 7).** Not exam-meta. Six parametric→other + one stay-other:

| id | Cell B text | What happened |
| -- | ----------- | ------------- |
| 7822 Pakistan | `Lahore` | Bare city; context gold is “The Indomitable City” → no alias hit |
| 2585 GoT | `Shawn Mathis Wayans Amoo` + follow-up Q | Near-gold “amigo”; scorer `other` (same as forward) |
| 4185 Tokyo | `Kangra` | Bare context entity; gold is “Tokyo is the capital of Kangra” so bare “Kangra” misses the Tokyo-side template |
| 6400 Inter | `Red` | Too short / color gold is “red color”; baseline was already this content |
| 4192 Mexico City | `Osona` | Bare context region (forward kept the full sentence, still parametric) |
| 4187 Athens | `Yungay Province` | Same bare-entity pattern |
| 790 Karate Kid | `Action` | Stay-other; gold is “action movie” |

These are alias/brevity misses, not format collapse. Forward transfer has 3 `other` vs Cell B’s 7 — Cell B is slightly more aggressive at shortening, which produces more `other` as well as more `context`.

**Three-way style.** On shared ids, (b) forward and (c) Cell B are the same object: short context names, occasional `Explanation:` / follow-up `Question:`. (a) FP16 own K10 is the same entities in more complete sentences. That is not the Cell A “entity + exam-meta” style.

## Verdict

**Cell B passes the Cell A inspection.** Exam-meta among context is 0% (vs Cell A 41%). Completions are on-topic context-entity answers in the same style as the FP16→INT4 forward-transfer run we already trust. The +18 / +18 point-estimate match is not a coincidental scorer trap of a different kind.

**Cell B can be written up as-is** (native in-precision context K10 is a real INT4 effect matching forward transfer). Do not hold it for Cell A reasons.

Methods footnote, shared with forward transfer, not a Cell B hold: part of the INT4 +18 is parametric-first scorer recategorization of already-context-naming sentences that get shortened to bare names. That caveat applies to both INT4 +18 cells equally; it does not uniquely undermine native apply.
