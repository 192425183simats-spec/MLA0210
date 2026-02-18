import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("energy_load.csv")

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

data["Load"] = le1.fit_transform(data["Load"])
data["Generation"] = le2.fit_transform(data["Generation"])
data["TimeSlot"] = le3.fit_transform(data["TimeSlot"])
data["Status"] = le4.fit_transform(data["Status"])

X = data[["Load", "Generation", "TimeSlot"]]
y = data["Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

new = pd.DataFrame({
    "Load": [le1.transform(["High"])[0]],
    "Generation": [le2.transform(["Low"])[0]],
    "TimeSlot": [le3.transform(["Peak"])[0]]
})

result = model.predict(new)
print("Prediction:", le4.inverse_transform(result))
