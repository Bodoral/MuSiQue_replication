# MuSiQue Replication
 python implementation of the paper [MuSiQue (Multihop Questions via Single-hop Question Composition)](https://arxiv.org/abs/2108.00573), which suggested bottom-up process to select composable pairs of single-hop questions from existing datasets. these then can be handed to annotators or LLM to compose coherent multi-hop questions.


## Get Started
1. clone the repo
2. cd to MuSiQue
3. build the image
```console
docker build --build-arg UID=$(id -u) -t musique:latest .
```
4. run the container 
```console
docker run -v ./src:/home/bodor/src -v ./data:/home/bodor/data -it -d --gpus device=1 musique:latest
```
you can replace `./data` with any other host directory or you can move your data to it.


5. attach to the container and run main.py
```
python3.11 src/main.py --in_files data/single_hop_questions.jsonl --out_path data/composable_questions
```


## Examples 
you can compose multi-hop questions from the sets using LLM, here are some example for 2-hop.  
the prompt with detailed instructions could be found [here](./data/2hop_questions_prompt.txt)
```
question 1: who ruined Al-Shifa hospital in Gaza? Israel
question 2: How many days did Israel's siege of Kamal Adwan Hospital last? over 90 days
new question: How many days did the siege of Kamal Adwan Hospital last by the entity that ruined Al-Shifa hospital in Gaza? over 90 days

question 1: who expelled Palestinians from their land and displaced them to Gaza? Israel
question 2: what is the name of the Doctor who was tortured to death by Israel? Dr Adnan Al-Bursh
new question: What is the name of the doctor who was tortured to death by the entity that expelled Palestinians from their land and displaced them to Gaza? Dr Adnan Al-Bursh

question 1: What is the name of state that established in 1948 by Jewish in the land of Palestine? Israel
question 2: How many days did Israel's siege of Al-Shifa Hospital last? 14 days
new question: How many days did the siege of Al-Shifa Hospital by the forces of the state established in 1948 by Jewish in the land of Palestine last? 14 days

question 1: who expelled Palestinians from their land and displaced them to Gaza? Israel
question 2: When Israel fully occupied Jerusalem? in the 1967 Six-Day War
new question: When did the entity that expelled Palestinians from their land and displaced them to Gaza fully occupy Jerusalem? in the 1967 Six-Day War

question 1: who burned Kamal Adwan hospital in Gaza? Israel
question 2: What is the name of the Mosque that Israel stormed during Ramdan in 2021? Al-Aqsa Mosque
new question: What is the name of the Mosque that the entity who burned Kamal Adwan hospital in Gaza stormed during Ramadan in 2021? Al-Aqsa Mosque

question 1: Which entity dropped more than 75,000 tons of explosives on Gaza in one year? Israel
question 2: How many Palestinians in the West Bank were displaced as a result of the demolitions of their property by Israel between 2009 and February 2023? 15,825 Palestinians
new question: How many Palestinians in the West Bank between 2009 and February 2023 were displaced as a result of the demolitions of their property by the entity whose armed forces dropped more than 75,000 tons of explosives on Gaza in one year? 15,825 Palestinians

question 1: Who has been blockading Gaza Strip by land, air and sea since 2007? Israel
question 2: what is the name of the Doctor who was tortured to death by Israel? Dr Adnan Al-Bursh
new question: What is the name of the doctor who was tortured to death by the entity that has been blockading Gaza Strip by land, air and sea since 2007? Dr Adnan Al-Bursh

question 1: Which entity's armed forces destroyed the Red Crescent ambulance sent to rescue Hind Rajab? Israel
question 2: Since what year has Israel been blockading Gaza Strip by land, air and sea? 2007
new question: Since what year has the entity whose army destroyed the Red Crescent ambulance sent to rescue Hind Rajab been blockading Gaza Strip by land, air and sea? 2007

question 1: Who colonized Palestine in the period between 1923 and 1948? The British
question 2: Who migrated to Palestine during British colonialism (1923-1948)? Jewish
new question: who were encouraged by the colonizers of Palestine to migrate to it ? Jewish

question 1: Where was the kid Rami Al-Halhouli shot dead? Jerusalem
question 2: Who built The Dome of the Rock in Jerusalem? Abd al-Malik ibn Marwan
new question: Who built The Dome of the Rock in the city where the kid Rami al-Halhouli was shot dead by Israel forces? Abd al-Malik ibn Marwan
```

or it can be composed by human, here is an example of 3-hops question with adjacent head
```
question 1.1: Where was the kid Rami Al-Halhouli shot dead? Jerusalem
question 1.2: who arrested the Director of Kamal Adwan Hospital? Israel
question 2: When did Israel occupied Jerusalem's western part? in 1948

new question: When did the entity whose forces arrested Director of Kamal Adwan Hospital occupied the western part of the city where the kid Rami al-Halhouli was shot dead? in 1948


```
and another for 3-hop question
```
question 1: Who is the fifth king of the Umayyad Arab dynasty? Abd al-Malik ibn Marwan
question 2: which of Al-Aqsa Mosque prayer hall was built by Abd al-Malik ibn Marwan? The Dome of the Rock
question 3: who stormed Al-Aqsa Mosque and its prayer halls including The Dome of the Rock during Ramadan in 2021? Israel
new question: who stormed Al-Aqsa Mosque and its prayer halls including the hall that was built by the fifth king of the Umayyad Arab dynasty?  Israel
```


## Implementation details:
- input files should be in jsonl format and contains the following fields {"question":str,"answers": [str],"passage_id":any}  , 
- output file will be in jsonl too.
- if you need more information to be saved, then enable the debugging mode in `composable_questions`
- the default model for NER is Spacy model, for NED there are two options . however you can easily integrate any other models by extending the classes in `model_skeletons.py`

## Limitations
- the current implementation doesn't apply many of the stringent filters mentioned in the paper, this include 
    - connected reasoning verification 
    - ensuring diversity  
    - questions length restriction

- creation of unanswerable questions is not implemented 

