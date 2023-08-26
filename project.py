import pandas as pd

# Import Data
cust = pd.read_csv('Dataset/Customer.csv', sep = ';')
prod = pd.read_csv('Dataset/Product.csv', sep = ';')
store = pd.read_csv('Dataset/Store.csv', sep = ';')
trans = pd.read_csv('Dataset/Transaction.csv', sep = ';')

# Mengubah tipe data incone menjadi int
cust['Income'] = cust['Income'].astype(str)
cust['Income'] = cust['Income'].str.replace(',', '')
cust['Income'] = cust['Income'].astype(int)

# Mengubah value gender dari 0 menjadi wanita, 1 menjadi pria
gender_map = {0: 'Wanita', 1 : 'Pria'}
cust['Gender'] = cust['Gender'].map(gender_map)

# Mengisi Marital Status yang kosong
mode_marital = cust['Marital Status'].mode()[0]

# Mengisi nilai kosong dengan mode
cust['Marital Status'].fillna(mode_marital, inplace=True)

# Menambahkan angka 0000 pada income agar nilainya presisi (jutaan)
cust['Income'] = cust['Income'].apply(lambda x: f"{x}0000")

# Mengubah tipe data Income menjadi int
cust['Income'] = cust['Income'].astype('int64')

# Mengubah tipe data date dari object menjadi datetime
trans['Date'] = pd.to_datetime(trans['Date'], format='%d/%m/%Y')

# Menggabungkan 4 dataset menjadi 1
df = trans.merge(prod, on='ProductID', how='left') \
          .merge(store, on='StoreID', how='left') \
          .merge(cust, on='CustomerID', how='left')

# Mengubah nama kolom price_x menjadi price
df.rename(columns={'Price_x' : 'Price'}, inplace = True)

# Menghapus kolom yang tidak perlu
df = df.drop(columns=['Price_y', 'Latitude', 'Longitude'])

# Time Series Regression Using ARIMA
# Groupby dan agregasi data berdasarkan tanggal
daily_qty = df.groupby('Date')['Qty'].sum()

# Memastikan tanggal dalam urutan waktu yang benar
daily_qty = daily_qty.sort_index()

# Menentukan frekuensi harian
daily_qty.index.freq = 'D'

# Memisahkan data menjadi data latihan dan data uji
train_size = int(0.8 * len(daily_qty))
train_data = daily_qty[:train_size]
test_data = daily_qty[train_size:]

from statsmodels.tsa.arima.model import ARIMA

# Membuat model ARIMA
order = (5, 1, 1)  # Nilai ini dapat disesuaikan
model = ARIMA(train_data, order=order)
model_fit = model.fit()

# Melakukan prediksi pada data uji
predictions = model_fit.forecast(steps=len(test_data))

import matplotlib.pyplot as plt

# Menampilkan hasil prediksi
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.title('ARIMA Time Series Regression')
plt.legend()
plt.show()

# Clustering
# Grouping berdasarkan CustomerID
aggregation = {
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}

# Agregasi Data
cluster_data = df.groupby('CustomerID').agg(aggregation).reset_index()

# Menghilangkan kolom CustomerID untuk keperluan clustering
X = cluster_data.drop('CustomerID', axis=1)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster yang diinginkan
num_clusters = 3

# Membuat objek KMeans dengan n_init yang ditetapkan
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)

# Melakukan proses clustering dengan KMeans
cluster_labels = kmeans.fit_predict(X_scaled)

# Menambahkan kolom cluster_labels ke dalam cluster_data
cluster_data['Cluster'] = cluster_labels

import matplotlib.pyplot as plt

# Visualisasi hasil clustering menggunakan scatter plot
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']  # Sesuaikan jumlah cluster dengan warna yang diinginkan

for cluster_num, color in zip(range(num_clusters), colors):
    cluster_points = cluster_data[cluster_data['Cluster'] == cluster_num]
    plt.scatter(cluster_points['Qty'], cluster_points['TotalAmount'], color=color, label=f'Cluster {cluster_num}')

plt.xlabel('Qty')
plt.ylabel('Total Amount')
plt.title('Clustering Results')
plt.legend()
plt.show()