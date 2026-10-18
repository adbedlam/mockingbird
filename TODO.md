# TODO

 - [ ] Do feature extraction
   - [x] Emoji
   - [ ] Emotion
   - [x] Leghth
   - [ ] Lexicon
   - [ ] Messaging manner
   - [ ] Temporal
 - [ ] Train a model or a few to predict the features of the response (gradient boosting probably).
Over the last k messages
    Topic (OHE)                      |               | Probability to use emoji
    Lengths of messages (int)        |——————————————>| Emotion of the response
    Belong to self (bool)            |               | Length of the response
    Emotions (Circumplex model pair) |
 - [ ] Create a harness to insert a system message based on the model's output