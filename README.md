# covid-19-chatbot-engine
Covid-19 Chatbot engine to decode question and reply with context

# Depedencies
```
tensorflow
tensorflow_hub
tensorflow_text
pandas
```

# Usage
```
from covid_nlp import covid
answer, confidence = covid.predict("what is coronavirus?")
print(answer, confidence)
```