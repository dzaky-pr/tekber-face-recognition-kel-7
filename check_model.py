
from tensorflow.keras.models import load_model

# Memuat model
model = load_model('model/Gender_model.h5')

# Melihat struktur model
model.summary()

# Menggunakan model untuk prediksi
predictions = model.predict(data_preprocessed)

