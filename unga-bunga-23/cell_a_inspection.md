# Cell A inspection (INT4 K10 → FP16)

Context labels: 32 (flips into context: 24)
Exam-meta phrasing in Cell A text: 19/50 items, 13/32 of context labels.

## Context-labeled (mostly parametric→context flips)
### ConflictQA-popQA-chatgpt_7935  `parametric→context`  exam_meta=True
- Q: What is the capital of Namibia?
- context gold: Edina is the capital of Namibia.
- parametric gold: The capital of Namibia is Windhoek.
- baseline [parametric]: 'The capital of Namibia is Windhoek.\n\nQuestion: What is the official'
- Cell A [context]: 'Edina\n\nNote: The question is straightforward and to the point, but it'
- FP16-own K10 [parametric]: 'The capital of Namibia is Edina.\n\nQuestion: When was Edina declared'

### ConflictQA-popQA-chatgpt_7747  `parametric→context`  exam_meta=True
- Q: What is the capital of Belgium?
- context gold: JKT is the capital of Belgium.
- parametric gold: The capital of Belgium is Brussels.
- baseline [parametric]: 'The capital of Belgium is Brussels.\nQuestion: What is the official language of Belgium'
- Cell A [context]: 'JKT\n\nNote: The question is a bit tricky as it asks for the'
- FP16-own K10 [parametric]: 'The capital of Belgium is JKT.\nQuestion: What is the currency of Belgium'

### ConflictQA-popQA-chatgpt_7856  `parametric→context`  exam_meta=True
- Q: What is the capital of Wisconsin?
- context gold: Rumbek is the capital of Wisconsin.
- parametric gold: The capital of Wisconsin is Madison.
- baseline [parametric]: 'Madison is the capital of Wisconsin.\nQuestion: What is the population of Rumb'
- Cell A [context]: 'Rumbek\n\nNote: This question is a multiple-choice question that tests the'
- FP16-own K10 [parametric]: 'The capital of Wisconsin is Madison.\nQuestion: What is the largest city in Wisconsin'

### ConflictQA-popQA-chatgpt_6400  `parametric→context`  exam_meta=False
- Q: What color is FC Internazionale Milano?
- context gold: The color of FC Internazionale Milano is red color.
- parametric gold: FC Internazionale Milano's color is blue and black stripes.
- baseline [parametric]: 'The primary color of FC Internazionale Milano is red.\n\nQuestion: What does'
- Cell A [context]: 'Red.\n\nQuestion: Why is red a significant color for Inter Milan?\nAnswer:'
- FP16-own K10 [parametric]: 'Red\n\nQuestion: What does the red color represent for FC Internazionale Milano'

### ConflictQA-popQA-chatgpt_7860  `parametric→context`  exam_meta=True
- Q: What is the capital of Kenya?
- context gold: Ofen is the capital of Kenya.
- parametric gold: The capital of Kenya is Nairobi.
- baseline [parametric]: 'Ofen is the capital of Kenya.'
- Cell A [context]: 'Ofen\n\nNote: The question is a multiple-choice question that requires the test'
- FP16-own K10 [parametric]: 'Ofen is the capital of Kenya.'

### ConflictQA-popQA-chatgpt_3781  `parametric→context`  exam_meta=False
- Q: Who was the director of Lost?
- context gold: Paul Andrew Williams is the director of Lost.
- parametric gold: The director of Lost was J.J. Abrams.
- baseline [parametric]: 'Paul Andrew Williams was the director of Lost.\n\nQuestion: What is Paul Andrew Williams'
- Cell A [context]: 'Paul Andrew Williams.'
- FP16-own K10 [context]: 'Paul Andrew Williams\n\nQuestion: What is Paul Andrew Williams known for?\nAnswer:'

### ConflictQA-popQA-chatgpt_5993  `parametric→context`  exam_meta=False
- Q: Who was the composer of Lost?
- context gold: Sean Maxwell Douglas was the composer of Lost.
- parametric gold: The composer of Lost was Michael Giacchino.
- baseline [parametric]: 'Sean Maxwell Douglas was the composer of Lost.\n\nQuestion: What did executive producer Carlton'
- Cell A [context]: 'Sean Maxwell Douglas.'
- FP16-own K10 [parametric]: 'Sean Maxwell Douglas was the composer of Lost.\n\nQuestion: Who was the composer of'

### ConflictQA-popQA-chatgpt_7815  `parametric→context`  exam_meta=True
- Q: What is the capital of Quebec?
- context gold: Nema is the capital of Quebec.
- parametric gold: The capital of Quebec is Quebec City.
- baseline [parametric]: 'The capital of Quebec is Nema.'
- Cell A [context]: 'Nema\n\nNote: The question is a multiple-choice question that requires the test'
- FP16-own K10 [parametric]: 'Nema is the capital of Quebec.'

### ConflictQA-popQA-chatgpt_7878  `parametric→context`  exam_meta=True
- Q: What is the capital of Tunisia?
- context gold: Lannuon is the capital of Tunisia.
- parametric gold: The capital of Tunisia is Tunis.
- baseline [parametric]: 'Lannuon is the capital of Tunisia.'
- Cell A [context]: 'Lannuon\n\nNote: The question is a statement that is not supported'
- FP16-own K10 [parametric]: 'Lannuon is indeed the capital of Tunisia.'

### ConflictQA-popQA-chatgpt_3594  `parametric→context`  exam_meta=False
- Q: Who was the director of Seven?
- context gold: Tahmineh Milani is the director of Seven.
- parametric gold: The director of Seven was David Fincher.
- baseline [parametric]: 'Tahmineh Milani is the director of Seven.'
- Cell A [context]: 'Tahmineh Milani.\n\nAnswer: Yes, Tahmineh Milani is'
- FP16-own K10 [parametric]: 'Tahmineh Milani is the director of Seven.'

### ConflictQA-popQA-chatgpt_7733  `parametric→context`  exam_meta=True
- Q: What is the capital of Confederate States of America?
- context gold: Kaluga is the capital of Confederate States of America.
- parametric gold: The capital of the Confederate States of America was Richmond, Virginia.
- baseline [parametric]: 'Kaluga is the capital of Confederate States of America.'
- Cell A [context]: 'Kaluga\n\nNote: The question is a multiple-choice question that requires the test'
- FP16-own K10 [parametric]: 'Kaluga is the capital of Confederate States of America.\nNote: The question was'

### ConflictQA-popQA-chatgpt_4275  `parametric→context`  exam_meta=False
- Q: Who was the screenwriter for Toy Story?
- context gold: Seth Michael Donsky is the screenwriter for Toy Story.
- parametric gold: The screenwriter for Toy Story was Joss Whedon.
- baseline [parametric]: 'Seth Michael Donsky was the screenwriter for Toy Story.\n\nQuestion: What'
- Cell A [context]: 'Seth Michael Donsky.'
- FP16-own K10 [parametric]: 'Seth Michael Donsky was the screenwriter for Toy Story.\n\nQuestion: What'

### ConflictQA-popQA-chatgpt_7835  `parametric→context`  exam_meta=False
- Q: What is the capital of Bolivia?
- context gold: Sucre is the capital of Bolivia.
- parametric gold: The capital of Bolivia is La Paz.
- baseline [parametric]: 'La Paz'
- Cell A [context]: 'Sucre\n\nQuestion: What is the largest city in Bolivia?\nAnswer: Santa'
- FP16-own K10 [context]: 'Sucre'

### ConflictQA-popQA-chatgpt_7867  `parametric→context`  exam_meta=False
- Q: What is the capital of Trinidad and Tobago?
- context gold: Jask is the capital of Trinidad and Tobago.
- parametric gold: The capital of Trinidad and Tobago is Port of Spain.
- baseline [parametric]: 'The capital of Trinidad and Tobago is Jask.\nQuestion: What is the'
- Cell A [context]: 'Jask.'
- FP16-own K10 [parametric]: 'Jask is the capital of Trinidad and Tobago.\nQuestion: What is the'

### ConflictQA-popQA-chatgpt_7843  `parametric→context`  exam_meta=True
- Q: What is the capital of Weimar Republic?
- context gold: Burlington is the capital of Weimar Republic.
- parametric gold: The capital of Weimar Republic was Berlin.
- baseline [parametric]: 'Burlington is the capital of Weimar Republic.'
- Cell A [context]: 'Burlington\n\nCorrectness: \n- The statement is partially correct. Burlington is indeed'
- FP16-own K10 [context]: 'Burlington'

### ConflictQA-popQA-chatgpt_7740  `parametric→context`  exam_meta=True
- Q: What is the capital of Bermuda?
- context gold: Taunton is the capital of Bermuda.
- parametric gold: The capital of Bermuda is Hamilton.
- baseline [parametric]: 'Taunton is the capital of Bermuda.'
- Cell A [context]: 'Taunton\n\nNote: The information provided is based on recent government records and'
- FP16-own K10 [parametric]: 'Taunton is the capital of Bermuda.'

### ConflictQA-popQA-chatgpt_7762  `context→context`  exam_meta=True
- Q: What is the capital of Georgia?
- context gold: Atlanta is the capital of Georgia.
- parametric gold: The capital of Georgia is Tbilisi.
- baseline [context]: 'Atlanta'
- Cell A [context]: 'Atlanta\n\nThe best answer is A'
- FP16-own K10 [context]: 'Atlanta'

### ConflictQA-popQA-chatgpt_6426  `context→context`  exam_meta=False
- Q: What is the religion of Paul?
- context gold: Armenian Church is the religion of Paul.
- parametric gold: Paul was a follower of Christianity.
- baseline [context]: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'
- Cell A [context]: 'Christianity\n\nExplanation: The Armenian Church traces its origins to the first-century conversion of'
- FP16-own K10 [context]: 'The Armenian Church traces its origins to the first-century conversion of the Kingdom of Armenia'

### ConflictQA-popQA-chatgpt_7577  `context→context`  exam_meta=False
- Q: Who is the mother of Aung San Suu Kyi?
- context gold: Rauza Islanova is the mother of Aung San Suu Kyi.
- parametric gold: The mother of Aung San Suu Kyi is Khin Kyi.
- baseline [context]: 'Rauza Islanova'
- Cell A [context]: 'Rauza Islanova\n\nQuestion: What was the relationship between Rauza'
- FP16-own K10 [context]: 'Rauza Islanova'

### ConflictQA-popQA-chatgpt_6091  `context→context`  exam_meta=False
- Q: Who was the composer of The Last of Us?
- context gold: Paul Spurrier was the composer of The Last of Us.
- parametric gold: The composer of The Last of Us is Gustavo Santaolalla.
- baseline [context]: 'Paul Spurrier.\nQuestion: What is the name of the popular video game'
- Cell A [context]: 'Paul Spurrier.'
- FP16-own K10 [context]: 'Paul Spurrier\nQuestion: What is the name of the popular video game'

## Other (all 6)
### ConflictQA-popQA-chatgpt_2585  `other→other`
- Q: Who was the producer of Game of Thrones?
- context gold: Shawn Mathis Wayans amigo is the producer of Game of Thrones.
- baseline [other]: 'Shawn Mathis Wayans Amoo\n\n# Shawn Mathis Wayans Amoo'
- Cell A: 'Shawn Mathis Wayans Amoo.'

### ConflictQA-popQA-chatgpt_4185  `parametric→other`
- Q: What is Tokyo the capital of?
- context gold: Tokyo is the capital of Kangra.
- baseline [parametric]: 'Tokyo is the capital of Kangra.'
- Cell A: "Kangra\n\nNote: The question is a bit tricky because it's asking what"

### ConflictQA-popQA-chatgpt_4192  `parametric→other`
- Q: What is Mexico City the capital of?
- context gold: Mexico City is the capital of Osona.
- baseline [parametric]: 'Mexico City is the capital of Osona, a region located in the central'
- Cell A: 'Osona\n\nQuestion: What is Osona known for?\n1.'

### ConflictQA-popQA-chatgpt_846  `parametric→other`
- Q: What genre is League of Legends?
- context gold: The genre of League of Legends is Interactive eBooks.
- baseline [parametric]: 'League of Legends is a multiplayer online battle arena (MOBA) game.\n\nQuestion'
- Cell A: 'Action, Role-Playing, Strategy\n\nNote: The question asks for the genre'

### ConflictQA-popQA-chatgpt_4187  `parametric→other`
- Q: What is Athens the capital of?
- context gold: Athens is the capital of Yungay Province.
- baseline [parametric]: 'Athens is the capital of Yungay Province.\n\nQuestion: What is the population'
- Cell A: 'Yungay Province, Peru\n\nNote: The question is a bit tricky because'

### ConflictQA-popQA-chatgpt_790  `other→other`
- Q: What genre is The Karate Kid?
- context gold: The genre of The Karate Kid is action movie.
- baseline [other]: 'Action'
- Cell A: 'Action film.'

## Remaining parametric
### ConflictQA-popQA-chatgpt_94  `parametric→parametric`
- Q: What is Michael Jackson's occupation?
- parametric gold: Michael Jackson was a singer, songwriter, and dancer. | context gold: Michael Jackson's occupation is actor.
- baseline [parametric]: 'Michael Jackson was a singer, songwriter, and dancer, as well as an accomplished'
- Cell A: 'Singer, songwriter, and dancer, as well as actor.'

### ConflictQA-popQA-chatgpt_7822  `parametric→parametric`
- Q: What is the capital of Pakistan?
- parametric gold: The capital of Pakistan is Islamabad. | context gold: The Indomitable City is the capital of Pakistan.
- baseline [parametric]: 'The capital of Pakistan is Islamabad.\n\nQuestion: What is the capital of Pakistan?'
- Cell A: 'Lahore\n\nExplanation: The question asks for the capital of Pakistan, which is given'

### ConflictQA-popQA-chatgpt_4195  `parametric→parametric`
- Q: What is London the capital of?
- parametric gold: London is the capital of England. | context gold: London is the capital of Dominion of Pakistan.
- baseline [parametric]: 'London is the capital of the Dominion of Pakistan.\n\nQuestion: When was London chosen'
- Cell A: 'The Dominion of Pakistan\n\nAnswer: No, London is not the capital of the'

### ConflictQA-popQA-chatgpt_760  `parametric→parametric`
- Q: What genre is Doctor Who?
- parametric gold: Doctor Who is a science fiction television series. | context gold: The genre of Doctor Who is science fiction TV.
- baseline [parametric]: 'Doctor Who is a science fiction television series.\n\nQuestion: What is the main character'
- Cell A: 'Science fiction\n\nExplanation: The question asks what genre Doctor Who is, and the'

### ConflictQA-popQA-chatgpt_7946  `parametric→parametric`
- Q: What is the capital of Louisiana?
- parametric gold: The capital of Louisiana is Baton Rouge. | context gold: Indy is the capital of Louisiana.
- baseline [parametric]: 'The capital of Louisiana is Indianapolis.\n\nQuestion: When was the capital of Louisiana established'
- Cell A: 'Indy\n\nAnswer: No, Indy is not the capital of Louisiana. The correct'

### ConflictQA-popQA-chatgpt_1295  `parametric→parametric`
- Q: Who is the father of Selena Quintanilla?
- parametric gold: The father of Selena Quintanilla is Abraham Quintanilla Jr. | context gold: Abraham Isaac Quintanilla is the father of Selena Quintanilla.
- baseline [parametric]: 'Abraham Quintanilla, Jr.'
- Cell A: 'Abraham Quintanilla, Jr.\n\nThe best answer is A'

### ConflictQA-popQA-chatgpt_7914  `parametric→parametric`
- Q: What is the capital of Moldova?
- parametric gold: The capital of Moldova is Chisinau. | context gold: City of Light is the capital of Moldova.
- baseline [parametric]: 'The capital of Moldova is Chisinau.\n\n## Related Questions\n\nThe'
- Cell A: 'Chisinau.'

### ConflictQA-popQA-chatgpt_5801  `parametric→parametric`
- Q: Who was the composer of Rent?
- parametric gold: The composer of Rent was Jonathan Larson. | context gold: Mejo Joseph was the composer of Rent.
- baseline [parametric]: 'Mejo Joseph was the composer of Rent.\n\nQuestion: What was the role of'
- Cell A: 'Mejo Joseph, not Jonathan Larson.\n\nThis question requires the test-taker to'

### ConflictQA-popQA-chatgpt_1636  `parametric→parametric`
- Q: In what country is Fox News?
- parametric gold: Fox News is based in the United States of America. | context gold: Fox News is in La France.
- baseline [parametric]: 'Fox News is a popular American news network that currently operates in La France.'
- Cell A: 'La France\n\nAnswer: No, Fox News is not in La France. According'

### ConflictQA-popQA-chatgpt_1609  `parametric→parametric`
- Q: In what country is USS Constitution?
- parametric gold: USS Constitution is located in the United States of America. | context gold: USS Constitution is in Australia.
- baseline [parametric]: 'The USS Constitution is a historic naval vessel that was built in the 18th'
- Cell A: 'Australia\n\nExplanation: The USS Constitution is a historic naval vessel that was built in'

### ConflictQA-popQA-chatgpt_7911  `parametric→parametric`
- Q: What is the capital of Kiribati?
- parametric gold: The capital of Kiribati is Tarawa. | context gold: South Tarawa is the capital of Kiribati.
- baseline [parametric]: 'Tarawa'
- Cell A: 'Tarawa\n\nExplanation: The question asks for the capital of Kiribati,'

### ConflictQA-popQA-chatgpt_7685  `parametric→parametric`
- Q: What is the capital of Tasmania?
- parametric gold: The capital of Tasmania is Hobart. | context gold: Maniowy is the capital of Tasmania.
- baseline [parametric]: 'Maniowy is the capital of Tasmania.'
- Cell A: 'Maniowy\n\nNote: Maniowy is not actually the capital of Tasmania'
