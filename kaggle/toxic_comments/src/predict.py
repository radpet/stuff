import pandas as pd

from util import load_trained_model, ys, load_clean_test, TEXT


def predict_test():

    test = load_clean_test()
    print('Test shape', test.shape)
    preds = {}
    for y in ys:
        print('Predicting for',y)
        model = load_trained_model(y+'_TF_IDF_NB_SENTI'+'_ft')
        probs = model.predict_proba(test[TEXT].values)
        preds[y] = model.predict_proba(probs)[:, 1]

    subm = pd.DataFrame(preds, index=test['id'])

    subm.to_csv('submission.csv')

if __name__ == '__main__':
    predict_test()
