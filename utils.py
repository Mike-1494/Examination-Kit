import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class CheatingDetection:
    def __init__(self):
        #self.model = RandomForestClassifier(n_estimators = 20, criterion = 'gini', random_state = 0, max_depth=8)
        self.model = SVC(kernel='rbf', random_state=0, C=1.0, gamma='scale')
    
    def load_data(self, csv_file):
        dataset = pd.read_csv(csv_file)
        self.y = dataset['label']
        self.X = dataset.iloc[:, 2:]
    
    def train_with_data(self):
        self.model.fit(self.X, self.y)
    
    def predict(self, nose_x, nose_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y, left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_index_finger_x, left_index_finger_y, right_index_finger_x, right_index_finger_y, left_eye_x, left_eye_y, left_eye_z, right_eye_x, right_eye_y, right_eye_z):
        predict_data = pd.DataFrame([(nose_x, nose_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y, left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_index_finger_x, left_index_finger_y, right_index_finger_x, right_index_finger_y, left_eye_x, left_eye_y, left_eye_z, right_eye_x, right_eye_y, right_eye_z)])
        y_pred = self.model.predict(predict_data)
        return y_pred[0]
    

if(__name__ == "__main__"):
    model  = CheatingDetection() 
    model.load_data('landmarks.csv')
    model.train_with_data()
    print(model.X)
    y = model.predict(0.39815, 0.23696, 0.64389, 0.24149, 0.18594, 0.26035, 100, 100, 0.13573, 0.44488, 100, 100, 0.37454, 0.2792, 100, 100, 0.42289, 0.22615, 0.22615, 0.21325, -1.19776, 0.44822, 0.2094, -1.19778)
    print(y)