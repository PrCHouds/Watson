import pandas as pd
from Config import *
submission = right_sub = pd.read_csv(DATA_DIR + "my_submission.csv")
answers = submission['prediction'].tolist()
print(answers)
right_sub = pd.read_csv(DATA_DIR + "submission.csv")
right_answers = right_sub['prediction'].tolist()
print(right_answers)
k = 0
for i in range(len(answers)):
    if answers[i] == right_answers[i]:
        k += 1
print(f'Right answers: {k} from {len(answers)}')
print(f'Accuracy: {k / len(answers) * 100}')
