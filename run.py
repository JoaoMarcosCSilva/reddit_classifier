import model, data, evaluate
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf

# Gets the MLP that will be trained on top of Bert
classifier = model.get_classifier(19)

# Load all the data (this may take a long time)
(x_train_df, y_train_df), (x_val_df, y_val_df), (x_test_df, y_test_df), df, labels = data.load_data()

# Instantiates Bert and its tokenizer
bert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Creates data generators for train, validation and test
train_gen = DataGenerator(x_train_df, y_train_df, steps = 50, batch_size = 64)
val_gen = DataGenerator(x_val_df, y_val_df, steps = 10, batch_size = 64)
test_gen = DataGenerator(x_test_df, y_test_df)

# Creates a list of all the label names in the one-hot encoding (used in the confusion matrix plot)
labels = list(y_train_df.columns)

# Compiles the MLP classifier
classifier.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True), 
        optimizer = tf.keras.optimizers.Adam(3e-4),
        metrics = ['categorical_accuracy'])

classifier.fit(train_gen, validation_data = val_gen, epochs = 30)

evaluate.plot_confusion_matrix(classifier, test_gen, 10, labels)
