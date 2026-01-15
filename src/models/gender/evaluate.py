from sklearn.metrics import accuracy_score


class GenderModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        return {"accuracy": accuracy}
