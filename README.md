# Skin Lesion Classification 


## İçindekiler

- [1: Cilt Lezyonu Sınıflandırma Projesi](#Cilt-Lezyonu-Sınıflandırma-Projesi)
- [2: Veri yolu tanımları](#Veri-yolu-tanımları)
- [3: Model Girişi ve Etiketleme](#Model-Girişi-Ve-Etiketleme)
- [4: Train Validation Örnek Sayıları ve Sınıf Mapping](#Train-Validation-Örnek-Sayıları-Ve-Sınıf-Mapping)
- [5: Örnek Batch Görselleştirme](#Örnek-Batch-Görselleştirme)
- [6: Augmentation Görsel Kontrol — Orijinal vs. Augmented](#Augmentation-Görsel-Kontrol--Orijinal-Vs-Augmented)
- [7: Hiperparametre Denemesi](#Hiperparametre-Denemesi)
- [8: CNN Baseline Modeli](#Cnn-Baseline-Modeli)
- [9: Sınıf Dağılımı ve Class Weight Analizi](#Sınıf-Dağılımı-Ve-Class-Weight-Analizi)
- [10: CNN Modeli Eğitimi ](#Cnn-Modeli-Eğitimi)
- [11: Overfitting Analizi](#Overfitting-Analizi)
- [12: Performans Metrikleri](#Performans-Metrikleri)
- [13: Confusion Matrix Analizi](#Confusion-Matrix-Analizi)
- [14: Classification Report Analizi](classification-report-analizi)
- [15: Değerlendirme](#cell-30-markdown-cell-30)
- [16: Model İyileştirmesi — EfficientNetB0 Transfer Learning](#cell-31-model-i̇yileştirmesi-efficientnetb0-transfer-learning)
- [17: MobileNetV2 Eğitim Sonuçları](#cell-33-mobilenetv2-eğitim-sonuçları)
- [18: CNN vs MobileNetV2 — Sonuç Karşılaştırması](#cell-35-cnn-vs-mobilenetv2-sonuç-karşılaştırması)
- [19: Grad-CAM Değerlendirmesi](#cell-37-grad-cam-değerlendirmesi)
- [20: Confusion Matrix (MobileNetV2)](#cell-39-confusion-matrix-mobilenetv2)
- [21: Classification Report MobileNetV2](#Classification-Report-MobileNetV2)
- [22: Final Özet](#Final-Özet)

---



---


#Cilt Lezyonu Sınıflandırma Projesi

## Veri Kümesi  
Bu projede **Skin Lesion Dataset** isimli Kaggle veri seti kullanılmıştır. Veri kümesinde **8 farklı sınıf** bulunmaktadır:  

- AK (Actinic Keratosis)  
- BCC (Basal Cell Carcinoma)  
- BKL (Benign Keratosis-like lesions)  
- DF (Dermatofibroma)  
- MEL (Melanoma)  
- NV (Melanocytic Nevi)  
- SCC (Squamous Cell Carcinoma)  
- VASC (Vascular lesions)  

Sınıfların örnek sayıları farklıdır; bu durum ciddi bir **sınıf dengesizliği** sorununa yol açmaktadır. Görseller dermoskopik cilt görüntülerinden oluşmaktadır.  

## Projenin Amacı  
Bu projenin amacı, cilt lezyonlarının doğru şekilde sınıflandırılmasıdır. Çalışma kapsamında:  

- İlk adımda basit bir **CNN modeli** kullanılmış, ancak düşük doğruluk ve **overfitting** gözlenmiştir.  
- Ardından **MobileNetV2** tabanı ile **transfer learning** uygulanmış, sonrasında **fine-tuning** adımı eklenerek daha dengeli bir model elde edilmiştir.  
- Eğitim sürecinde **class weight** kullanılarak dengesiz veri seti dengelenmeye çalışılmıştır.  

## Kullanılan Yöntemler  
- **CNN Baseline:** Referans performans için başlangıç modeli.  
- **MobileNetV2 (Transfer Learning + Fine-Tuning):** Daha yüksek doğruluk ve genelleme için.  
- **Callback’ler:** EarlyStopping, ReduceLROnPlateau ve ModelCheckpoint ile eğitim süreci optimize edilmiştir.  
- **Değerlendirme:** Accuracy/Loss grafikleri, Confusion Matrix, Classification Report ve Grad-CAM görselleştirmeleri.  
- **Hiperparametre Denemeleri:** Dropout ve öğrenme oranı üzerinde küçük çaplı testler.  

## Bulgular  
- **MobileNetV2**, CNN’e kıyasla belirgin şekilde daha yüksek doğruluk (**%54**) sağlamıştır.  
- Veri dengesizliği nedeniyle küçük sınıflarda başarı düşük kalmış, büyük sınıflarda (özellikle NV) daha başarılı sonuçlar alınmıştır.  
- **Grad-CAM** görselleri, modelin çoğunlukla lezyon bölgelerine odaklandığını, yanlış sınıflarda ise odağın dağınık olduğunu göstermiştir.  

Genel olarak, bu proje basit bir CNN’den transfer learning tabanlı bir modele geçişin, sınıflandırma performansını nasıl artırdığını göstermektedir.  

---


```python
#Gerekli kütüphaneler indirildi

import os, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn import metrics as M
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

---


**Veri Yolu Tanımları**

Kaggle ortamında yüklenen tüm veri setleri **`/kaggle/input/`** dizini altında saklanmaktadır.  
Bu projede kullanılan **Skin Lesion Dataset** için yol değişkenleri aşağıdaki gibi tanımlanmıştır:

- **`data_dir`** → Veri setinin ana klasörünü gösterir.  
  Örn: `/kaggle/input/skin-lesion-dataset`

- **`train_dir`** → Eğitim (Training) verilerinin bulunduğu klasör yolunu gösterir.  
  Örn: `/kaggle/input/skin-lesion-dataset/Train`

- **`val_dir`** → Doğrulama (Validation) verilerinin bulunduğu klasör yolunu gösterir.  
  Örn: `/kaggle/input/skin-lesion-dataset/Val`

Bu tanımlamalar sayesinde **`ImageDataGenerator.flow_from_directory()`** fonksiyonu ile ilgili klasörlerden görüntüler otomatik olarak okunabilir.Böylece eğitim ve doğrulama aşamalarında veri yükleme işlemi esnek ve hatasız bir biçimde gerçekleştirilmektedir.


---


```python
data_dir  = "/kaggle/input/skin-lesion-dataset"  
train_dir = os.path.join(data_dir, "Train")
val_dir   = os.path.join(data_dir, "Val")
```

---


## Model Girişi ve Etiketleme

Bu bölümde modelin giriş boyutunu ve veri etiketlerini tanımlıyoruz:

- **`input_shape = (224,224,3)`** → Model 64x64 boyutunda RGB (3 kanal) görüntüler bekliyor.  
- **`target_size = (224,224)`** → Data generator her resmi bu boyuta yeniden ölçekleyecek.  
- **`batch_size = 32`** → Eğitim sırasında aynı anda işlenecek örnek sayısı.  
- **`seed = 42`** → Rastgele işlemleri kontrol altına almak için, her çalıştırmada aynı sonuç alınmasını sağlayacak. 
- **`class_to_index`** → Sınıf isimlerini sayısal etiketlere (0–7) dönüştüren sözlük. CNN modeli kategorileri bu sayısal etiketlerle öğreniyor.

Model akışının sorusunsuz olması için input shape ve target_size eşitlenmiştir.
Bu adımda, deri lezyonu veri seti eğitim ve doğrulama alt kümelerine ayrılarak modele uygun biçimde hazırlanmıştır. Veri kümesi, ayrı bir Val klasörüne ihtiyaç duymadan, aynı Train klasörü içerisinden %80 eğitim ve %20 doğrulama olacak şekilde otomatik bölünmüştür.

Bu sayede her epoch sırasında veri hem normalize ediliyor hem de eğitim/doğrulama ayrımı otomatik yapılıyor.

---



```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = (224, 224, 3)
target_size = input_shape[:2]
batch_size  = 32
seed        = 42

# %20 validation split, aynı seed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.02,
    height_shift_range=0.02,
    zoom_range=0.05,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_dir = "/kaggle/input/skin-lesion-dataset/Train"  

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=seed
)

val_gen = val_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=False,
    seed=seed
)

# Kontrol: mapping aynı mı?
print("Train indices:", train_gen.class_indices)
print("Val   indices:", val_gen.class_indices)
#Eğitim ve doğrulama setlerinin aynı sınıf sıralamasına sahip olduğu kontrolü
assert train_gen.class_indices == val_gen.class_indices

print("Train samples:", train_gen.samples, "| Val samples:", val_gen.samples)
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
Found 22349 images belonging to 8 classes.
Found 5585 images belonging to 8 classes.
Train indices: {'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'SCC': 6, 'VASC': 7}
Val   indices: {'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'SCC': 6, 'VASC': 7}
Train samples: 22349 | Val samples: 5585
```


</details>

---


## Train & Validation Örnek Sayıları ve Sınıf Mapping

## Data Generator: Train (Normalize + Augmentation), Validation (Sadece Normalize)

- **Train Generator (`train_gen`)**  
  Eğitim verileri üzerinde hem **normalize** işlemi (`rescale=1./255`) hem de hafif **augmentation** (rotation, shift, zoom, horizontal flip) uygulanmaktadır.  
  Böylece model her epoch’ta aynı resmi farklı varyasyonlarla görür, bu da **genelleme gücünü artırır** ve **overfitting riskini azaltır**.  

- **Validation Generator (`val_gen`)**  
  Doğrulama verileri üzerinde yalnızca **normalize** işlemi yapılır.  
  Validation setinde augmentation uygulanmaz çünkü modelin gerçek performansını **değiştirilmemiş veriler** üzerinde görmek gerekir.  

👉 Sonuç: Eğitim sırasında çeşitlilik artırılırken, doğrulama süreci temiz tutulur.
weight_shift ve height_shit i 0.10 uyguladığımızda resimler bulanıklaştırılmıştır  ve .02 ye indirilmiştir.

---


```python
# === TRAIN: normalize + augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,           # normalize (kesinlikle gerekli)
    rotation_range=10,        # küçük rotasyon (±10° yeterli)
    width_shift_range=0.02,   # %2’den fazla kaydırma yapma
    height_shift_range=0.02,  # aynı şekilde %2 civarı
    zoom_range=0.05,          # %5 zoom → çok fazla yakınlaştırma/uzaklaştırma yok
    horizontal_flip=True,     # simetrik yapılar için faydalı
    vertical_flip=False,      # genelde deri görselleri ters çevrilmez, kapalı tut
    brightness_range=[0.9,1.1], # ışık koşullarını biraz değiştirmek faydalı
    validation_split=0.2)

# === VAL: sadece normalize ===
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,)

# Akışlar (Train klasöründen %80/%20)
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True,
    subset='training',
    seed=seed)

val_gen = val_datagen.flow_from_directory(
    train_dir,                 # validation split yine Train üzerinden
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False,
    subset='validation',
   
    seed=seed)

print("Train samples:", train_gen.samples, "| Val samples:", val_gen.samples)
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
Found 22349 images belonging to 8 classes.
Found 5585 images belonging to 8 classes.
Train samples: 22349 | Val samples: 5585
```


</details>

---


## Örnek Batch Görselleştirme

Bu adımda `train_gen` üzerinden bir batch (küme) veri çekilerek görselleştirme yapılmıştır.

- `next(train_gen)` ile eğitim generator’undan bir batch alınır.  
- `class_indices` kullanılarak sayısal etiketler sınıf isimlerine çevrilir (`idx_to_class`).  
- Görseller 3x3’lük bir grid halinde çizilir, başlıklarda ilgili sınıf adı gösterilir.  
- Amaç: Verilerin doğru şekilde okunup etiketlendiğini, augmentation sonrası görüntülerin anlamlı kaldığını kontrol etmektir.  

Bu kontrol, model eğitimine başlamadan önce veri pipeline’ının doğru çalıştığını **doğrulama** açısından kritik bir adımdır.


---


```python
import matplotlib.pyplot as plt
import numpy as np

# 1 batch çekelim
x_batch, y_batch = next(train_gen)

# Mapping (index → class adı)
idx_to_class = {v:k for k,v in train_gen.class_indices.items()}

# 9 görsel çizelim
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_batch[i])  # normalize edilmiş (0-1), direk gösterilebilir
    plt.title(idx_to_class[int(y_batch[i])])
    plt.axis('off')

plt.tight_layout()
plt.show()
```
<img width="766" height="790" alt="download" src="https://github.com/user-attachments/assets/2be19be5-71b9-43b1-b1b9-f6a2c855fd22" />


<details>
<summary><strong>Show cell outputs</strong></summary>

```text
<Figure size 800x800 with 9 Axes>
```

![output](readme_assets/cell11_out1.png)

</details>

---


## Augmentation Görsel Kontrol — Orijinal vs. Augmented

Amaç:
Eğitim sırasında uygulanan veri artırma tekniklerinin (rotation, shift, zoom, flip, brightness) görüntü kalitesini bozup bozmadığını incelemektir. Bu amaçla aynı görselin hem orijinal hali hem de augment edilmiş varyasyonları yan yana gösterilerek görsel kontrol yapılmıştır.

Yöntem:

**Orijinal Görsel:** Validation generator’dan seçilmiştir (validation aşamasında augmentation uygulanmadığından temiz görüntü elde edilir).

**Augment Edilmiş Görseller:** Eğitimde kullanılan parametrelerle oluşturulan bir ImageDataGenerator üzerinden flow(...) fonksiyonu yardımıyla üretilmiştir.

**Normalize:** Tüm görüntüler [0,1] aralığına normalize edilmiştir.

**Kenar Boşluklarının Önlenmesi:** Döndürme ve kaydırma işlemleri sonrası oluşabilecek siyah bölgeleri azaltmak için fill_mode="nearest" kullanılmıştır.

**Görselleştirme:** Orijinal görsel ve N adet augment edilmiş kopya aynı grid üzerinde yan yana gösterilmiştir.

**Sonuç:**
Bu kontrol sayesinde augmentasyon parametrelerinin görüntüleri aşırı derecede bozmadığı, lezyonun odakta kaldığı ve verinin tıbbi anlamının korunduğu doğrulanmıştır. Aşırı kaydırma veya zoom durumunda oluşabilecek bulanıklıklar fark edilerek parametreler daha uygun değerlere düşürülmüştür.

---


```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Orijinal resmi validation'dan al (augmentation yok)
idx = 0  
orig_path = val_gen.filepaths[idx]
orig_img  = load_img(orig_path, target_size=val_gen.target_size, color_mode="rgb")
orig_arr  = img_to_array(orig_img).astype("float32")  # 0..255 aralığı

# Eğitimdeki augment ayarlarıyla VIZ datagen (fill_mode ekledik)
viz_datagen = ImageDataGenerator(
    rescale=1./255,              # eğitimde olduğu gibi
    rotation_range=10,
    width_shift_range=0.02,
    height_shift_range=0.02,
    zoom_range=0.05,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode="nearest"          # siyah boşlukları önler
)

# Tek resmi batch'e sarıp flow ile N adet augment üret
N = 6
it = viz_datagen.flow(np.expand_dims(orig_arr, 0), batch_size=1, shuffle=False)
augmented = [next(it)[0] for _ in range(N)]   # her biri [0..1] aralığında

# Çizim (sol üstte orijinal, diğerleri augment)
plt.figure(figsize=(12,6))
plt.subplot(2,4,1)
plt.imshow(orig_arr / 255.0, vmin=0, vmax=1)  # orijinali 0..1'e getir
plt.title("Orijinal"); plt.axis("off")

for i, img in enumerate(augmented, start=2):
    plt.subplot(2,4,i)
    plt.imshow(np.clip(img, 0, 1), vmin=0, vmax=1)
    plt.title(f"Aug {i-1}")
    plt.axis("off")

plt.tight_layout(); plt.show()

# min/max değerleri
print("orig min/max:", float(orig_arr.min()), float(orig_arr.max()))
print("aug  min/max:", float(np.min([a.min() for a in augmented])),
                      float(np.max([a.max() for a in augmented])))
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
<Figure size 1200x600 with 7 Axes>
orig min/max: 49.0 255.0
aug  min/max: 0.1764705926179886 1.0
```

<img width="1223" height="581" alt="image" src="https://github.com/user-attachments/assets/98dbae52-b4dd-405f-9f88-a862b9749078" />

</details>

---


## Hiperparametre Denemesi

**Amaç:**
MobileNetV2 modelinde küçük ayarlarla (dropout oranı, öğrenme oranı, son katmanları açıp açmama) performansın nasıl değiştiğini görmek.

**Yöntem:**

**Dropout:** 0.3 ve 0.5 olarak denendi.

**Learning rate (lr):** 1e-3 ve 5e-4 olarak denendi.

**Son katmanları açma (unfreeze):** Sadece eklenen katman eğitildi ve son 10 katman açıldı.

Her deneme sadece **2 epoch** ile hızlıca test edildi.

Sonuçlar Accuracy, Macro F1 ve Balanced Accuracy metrikleriyle ölçüldü.

**Sonuç:**
Daha düşük dropout oranı (0.3), doğruluğu biraz daha yüksek verdi. Bu da modelin aşırı düzenlenmeden (over-regularization) daha iyi genelleme yaptığını gösteriyor. Daha uzun eğitimlerde de 0.3 değerinin daha uygun olacağı söylenebilir.

---


```python
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score

#Hızlı deneme fonksiyonu (dropout ve lr parametreleriyle)
def quick_try(dropout=0.3, lr=1e-3):
    # Önceden eğitilmiş MobileNetV2 tabanını alıyoruz
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=train_gen.image_shape, pooling="avg")
    base.trainable = False   # taban katmanlar donduruluyor (transfer learning)

#Model yapısı
    inp = layers.Input(shape=train_gen.image_shape)
    x = base(inp, training=False)
    x = layers.Dropout(dropout)(x)   # dropout uygulanıyor
    out = layers.Dense(train_gen.num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

#Hızlı eğitim (1 epoch, 50 step ile sınırlı)
    model.fit(
            train_gen, 
            validation_data=val_gen,
            epochs=1, 
            steps_per_epoch=50, 
            verbose=0)

    #Tahmin ve doğruluk hesabı
    y_pred = model.predict(val_gen, verbose=0).argmax(1)
    acc = accuracy_score(val_gen.classes, y_pred)
    return acc

#Dropout oranlarını karşılaştır (0.3 vs 0.5)
acc1 = quick_try(dropout=0.3)
acc2 = quick_try(dropout=0.5)

print(f"Dropout=0.3 | Accuracy={acc1:.3f}")
print(f"Dropout=0.5 | Accuracy={acc2:.3f}")
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
[1m9406464/9406464[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1758886007.409169      98 service.cc:148] XLA service 0x7a9a68003bf0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1758886007.409994      98 service.cc:156]   StreamExecutor device (0): Tesla T4, Compute Capability 7.5
I0000 00:00:1758886007.410017      98 service.cc:156]   StreamExecutor device (1): Tesla T4, Compute Capability 7.5
I0000 00:00:1758886008.349712      98 cuda_dnn.cc:529] Loaded cuDNN version 90300
I0000 00:00:1758886013.896741      98 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

Dropout=0.3 | Accuracy=0.479
Dropout=0.5 | Accuracy=0.491
```


</details>

---


## CNN Baseline Modeli

**Amaç:**
Transfer learning yöntemlerine geçmeden önce, sıfırdan tanımlanan bir CNN modeli ile referans (baseline) performansı elde etmek. Böylece EfficientNet gibi daha güçlü modellerle yapılacak karşılaştırmalar için sağlam bir başlangıç noktası oluşturmak.

Bu modelde üç konvolüsyon bloğu kullanılmıştır.Her bloktan sonra **Batch Normalization** ve **ReLU aktivasyonu** uygulanarak eğitim süreci daha kararlı hale getirilmiş, **MaxPooling** ile boyut küçültülmüş, **Dropout** ile aşırı öğrenme (overfitting) riski azaltılmıştır.

Son aşamada **Global Average Pooling (GAP)** ile çıkarılan özellikler özetlenmiş, ardından tam bağlı katmanlar aracılığıyla sınıflandırma yapılmıştır. Çıkış katmanı, tüm sınıflar için olasılık tahminleri üretmektedir.

**Sonuç:**
Bu CNN modeli, hafif ve hızlı bir yapı sunarak projede ilk performans ölçütü olmuştur. Baseline olarak elde edilen sonuçlar, daha sonra uygulanacak transfer learning (EfficientNetB0) ve fine-tuning adımlarının gelişimini değerlendirmek için bir karşılaştırma zemini sağlamaktadır.

---


```python
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

input_shape = train_gen.image_shape   # (224, 224, 3)
num_classes = train_gen.num_classes

def make_cnn_baseline(input_shape, num_classes):
    # === Input ===
    x = inp = layers.Input(shape=input_shape)

    # Blok 1 (düşük seviye özellikler)
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.15)(x)

    # Blok 2 (orta seviye özellikler)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.20)(x)

    # Blok 3 (yüksek seviye özellikler)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)

    # Flatten yerine GAP
    x = layers.GlobalAveragePooling2D()(x)      #Parametre sayısını azaltma
    x = layers.Dense(256, activation='relu')(x) # Tam bağlı katman
    x = layers.Dropout(0.3)(x)                  #Overfitting önleme
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    return models.Model(inp, out, name="cnn_baseline")
#Model derlenmesi
cnn = make_cnn_baseline(input_shape, num_classes)
cnn.compile(optimizer=Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.summary()
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
### CNN Baseline Model Summary

| Layer (type)                | Output Shape        | Param #  |
|------------------------------|---------------------|----------|
| **input_layer (InputLayer)** | (None, 224, 224, 3) | 0        |
| conv2d                       | (None, 224, 224, 32)| 864      |
| batch_normalization          | (None, 224, 224, 32)| 128      |
| re_lu                        | (None, 224, 224, 32)| 0        |
| conv2d_1                     | (None, 224, 224, 32)| 9,216    |
| batch_normalization_1        | (None, 224, 224, 32)| 128      |
| re_lu_1                      | (None, 224, 224, 32)| 0        |
| max_pooling2d                | (None, 112, 112, 32)| 0        |
| dropout                      | (None, 112, 112, 32)| 0        |
| conv2d_2                     | (None, 112, 112, 64)| 18,432   |
| batch_normalization_2        | (None, 112, 112, 64)| 256      |
| re_lu_2                      | (None, 112, 112, 64)| 0        |
| conv2d_3                     | (None, 112, 112, 64)| 36,864   |
| batch_normalization_3        | (None, 112, 112, 64)| 256      |
| re_lu_3                      | (None, 112, 112, 64)| 0        |
| max_pooling2d_1              | (None, 56, 56, 64)  | 0        |
| dropout_1                    | (None, 56, 56, 64)  | 0        |
| conv2d_4                     | (None, 56, 56, 128) | 73,728   |
| batch_normalization_4        | (None, 56, 56, 128) | 512      |
| re_lu_4                      | (None, 56, 56, 128) | 0        |
| conv2d_5                     | (None, 56, 56, 128) | 147,456  |
| batch_normalization_5        | (None, 56, 56, 128) | 512      |
| re_lu_5                      | (None, 56, 56, 128) | 0        |
| max_pooling2d_2              | (None, 28, 28, 128) | 0        |
| dropout_2                    | (None, 28, 28, 128) | 0        |
| global_average_pooling2d     | (None, 128)         | 0        |
| dense                        | (None, 256)         | 33,024   |
| dropout_3                    | (None, 256)         | 0        |
| dense_1                      | (None, 8)           | 2,056    |

---

**Total params:** 323,432 (1.23 MB)  
**Trainable params:** 322,536 (1.23 MB)  
**Non-trainable params:** 896 (3.50 KB)  

```


</details>

---


## Sınıf Dağılımı ve Class Weight Analizi

**Gözlem:**
Eğitim sürecinde modelin tahminlerinin büyük ölçüde NV (Nevus) sınıfına toplandığı fark edilmiştir. Bu durum, veri setindeki sınıf dengesizliğinden kaynaklanmaktadır.

**Eğitim Seti Dağılımı (Class counts):**  
- Sınıf 0 → 1.264 örnek  
- Sınıf 1 → 3.326 örnek  
- Sınıf 2 → 2.293 örnek  
- Sınıf 3 → 192 örnek  
- Sınıf 4 → 3.774 örnek  
- Sınıf 5 → 10.639 örnek (**en büyük sınıf**)  
- Sınıf 6 → 656 örnek  
- Sınıf 7 → 205 örnek  

👉 Görüldüğü gibi sınıflar arasında ciddi **dengesizlik** var. Özellikle sınıf 5 (10.639 örnek) çok baskınken, sınıf 3 (192 örnek) ve sınıf 7 (205 örnek) oldukça az temsil edilmiş.  

**Hesaplanan Class Weight değerleri:**  
- Küçük sınıflar:  
  - Sınıf 3 → **14.55**  
  - Sınıf 7 → **13.63**  
  - Sınıf 6 → **4.26**  
- Büyük sınıflar:  
  - Sınıf 5 → **0.263**  
  - Sınıf 4 → **0.74**  
  - Sınıf 1 → **0.84**  

👉 Yorum:  
- **Az örneği olan sınıflara yüksek ağırlık** verilerek modelin onları dikkate alması sağlanıyor.  
- **Çok örneği olan sınıflara düşük ağırlık** verilerek baskın olmaları engelleniyor.  
- Bu sayede eğitim sırasında model, her sınıfa daha **dengeli** yaklaşacak ve küçük sınıfları göz ardı etme ihtimali azalacak.


---

### Cell 19: Imports

```python
import numpy as np

# Toplam sınıf sayısı
num_classes = train_gen.num_classes

# Eğitim setindeki sınıf dağılımı
y_train = train_gen.classes
counts  = np.bincount(y_train, minlength=num_classes)
total   = y_train.size

# class_weight hesaplama: az sınıfa daha büyük ağırlık
class_weight = {
    i: total / (len(counts) * max(1, counts[i]))
    for i in range(num_classes)
}

print("Class counts :", dict(enumerate(counts)))
print("Class weight :", {k: round(v, 3) for k, v in class_weight.items()})
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
Class counts : {0: 1264, 1: 3326, 2: 2293, 3: 192, 4: 3774, 5: 10639, 6: 656, 7: 205}
Class weight : {0: 2.21, 1: 0.84, 2: 1.218, 3: 14.55, 4: 0.74, 5: 0.263, 6: 4.259, 7: 13.627}
```


</details>

---


## CNN Modeli Eğitimi

**Amaç:**
Cilt lezyonu sınıflandırma probleminde CNN tabanlı bir model eğitmek. Eğitim sırasında sınıf dengesizliğini gidermek için class_weight kullanılmakta, ayrıca modelin daha sağlıklı öğrenmesi için **EarlyStopping**, **ReduceLROnPlateau** ve **ModelCheckpoint** callback’leri uygulanmaktadır.

**Sonuç:**
- Callback’ler sayesinde model, gereksiz epoch’larda aşırı öğrenmeden korunmuş, en iyi genelleme performansına sahip olan noktadaki ağırlıklarla teslim edilmeye hazır hale gelmiştir.
- Model 20 epoch’a kadar eğitilmek üzere ayarlanmıştır, ancak callback mekanizmaları sayesinde daha erken sonlandırılabilmektedir.
- En düşük doğrulama kaybına sahip model /kaggle/working/cnn_baseline_best.keras dosyasına kaydedilmiştir.
- Eğitim süreci history_cnn değişkeninde saklanmıştır.
- Bu çıktı ilerleyen adımlarda metrik değerlendirme, confusion matrix ve hata analizi için kullanılacaktır.
- Modelin en iyi doğrulama performansı 1. epoch’ta elde edilmiştir.
- Eğitim, bu noktadan sonra gelişme olmadığı için durdurulmuş ve ağırlıklar otomatik olarak 1. epoch’taki en iyi haline geri yüklenmiştir.

---

```python
# Gerekli Kütüphaneler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models
import tensorflow as tf

# Mixed Precision Training (Hızlandırma için)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# EarlyStopping, ReduceLROnPlateau ve ModelCheckpoint Callback'leri
early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
ckpt  = ModelCheckpoint("/kaggle/working/cnn_baseline_best.keras", monitor="val_loss", save_best_only=True, verbose=1)

# CNN Modeli Tanımlama (Daha Basit ve Hızlı)
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),  # 128'lik konvolüsyonel katman yerine 64
        layers.Dense(64, activation='relu'),  # Daha küçük Dense katman
        layers.Dropout(0.3),  # Dropout oranını düşür
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Modeli Başlat
cnn = build_model()

# Eğitim Parametreleri
EPOCHS = 20

# Eğitimde Hızlandırma
history_cnn = cnn.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    batch_size=64,  # Daha büyük batch size
    callbacks=[early, rlrop, ckpt],
    class_weight=class_weight,  # Sınıf dengesizliğini gidermek için
    verbose=1
)
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
Epoch 1/20

2025-09-26 11:30:54.962898: E external/local_xla/xla/service/slow_operation_alarm.cc:65] Trying algorithm eng19{k2=0} for conv (f16[64,3,3,32]{3,2,1,0}, u8[0]{0}) custom-call(f16[32,111,111,32]{3,2,1,0}, f16[32,109,109,64]{3,2,1,0}), window={size=3x3}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convBackwardFilter", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]} is taking a while...
2025-09-26 11:30:55.194095: E external/local_xla/xla/service/slow_operation_alarm.cc:133] The operation took 1.23140252s
Trying algorithm eng19{k2=0} for conv (f16[64,3,3,32]{3,2,1,0}, u8[0]{0}) custom-call(f16[32,111,111,32]{3,2,1,0}, f16[32,109,109,64]{3,2,1,0}), window={size=3x3}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convBackwardFilter", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]} is taking a while...

[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 593ms/step - accuracy: 0.2930 - loss: 3.5452
Epoch 1: val_loss improved from inf to 1.64446, saving model to /kaggle/working/cnn_baseline_best.keras
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m444s[0m 621ms/step - accuracy: 0.2931 - loss: 3.5436 - val_accuracy: 0.3987 - val_loss: 1.6445 - learning_rate: 0.0010
Epoch 2/20
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402ms/step - accuracy: 0.3367 - loss: 1.9814
Epoch 2: val_loss did not improve from 1.64446
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m296s[0m 423ms/step - accuracy: 0.3367 - loss: 1.9813 - val_accuracy: 0.3332 - val_loss: 1.8528 - learning_rate: 0.0010
Epoch 3/20
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399ms/step - accuracy: 0.3399 - loss: 1.9202
Epoch 3: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.

Epoch 3: val_loss did not improve from 1.64446
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m294s[0m 421ms/step - accuracy: 0.3399 - loss: 1.9202 - val_accuracy: 0.2981 - val_loss: 1.7909 - learning_rate: 0.0010
Epoch 4/20
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 397ms/step - accuracy: 0.3620 - loss: 1.8264
Epoch 4: val_loss did not improve from 1.64446
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m294s[0m 420ms/step - accuracy: 0.3620 - loss: 1.8264 - val_accuracy: 0.3083 - val_loss: 1.8387 - learning_rate: 5.0000e-04
Epoch 5/20
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396ms/step - accuracy: 0.3624 - loss: 1.8095
Epoch 5: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.

Epoch 5: val_loss did not improve from 1.64446
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m293s[0m 419ms/step - accuracy: 0.3624 - loss: 1.8096 - val_accuracy: 0.2920 - val_loss: 1.8072 - learning_rate: 5.0000e-04
Epoch 5: early stopping
Restoring model weights from the end of the best epoch: 1.
```


</details>

---


## Overfitting Analizi

**Amaç:**  
CNN modelinde aşırı öğrenme olup olmadığını eğitim ve doğrulama eğrileri üzerinden görmek.  

**Sonuç:**  
- **Accuracy:** Eğitim doğruluğu %26’dan %40’a yükselirken, doğrulama doğruluğu dalgalı seyretti ve bazı epoch’larda %20’nin altına düştü. Bu durum modelin doğrulama setinde genelleme yapmakta zorlandığını gösteriyor.  
- **Loss:** Eğitim kaybı düzenli biçimde azalırken, doğrulama kaybı bir noktada yükselip tekrar düşmüş. Bu da doğrulama setinde kararsız bir öğrenme süreci yaşandığını gösteriyor.   

**Yorum:**  
Eğitim setinde iyileşme devam ederken doğrulama performansı geriledi. Bu rakamlar, modelin erken epoch’lardan itibaren **overfitting** yaptığını net biçimde gösteriyor.


---


```python
import matplotlib.pyplot as plt
h = history_cnn.history

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(h['accuracy']); plt.plot(h['val_accuracy'])
plt.title('CNN | Accuracy'); plt.legend(['train','val'])
plt.subplot(1,2,2); plt.plot(h['loss']); plt.plot(h['val_loss'])
plt.title('CNN | Loss'); plt.legend(['train','val'])
plt.tight_layout(); plt.show()
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
<Figure size 1000x400 with 2 Axes>
```

<img width="1038" height="396" alt="image" src="https://github.com/user-attachments/assets/8d4ac0f8-1f59-49c2-911b-5cee3a2b45f7" />


</details>

---


## Performans Metrikleri

**Amaç:**  
CNN modelinin doğrulama kümesi üzerindeki başarısını ölçmek. Burada sadece genel doğruluk değil, sınıf dengesizliğini dikkate alan **Macro F1** ve **Balanced Accuracy** metrikleri de kullanılmıştır.

**Sonuç:**  
Doğrulama kümesi üzerinde elde edilen skorlar:  

- **Accuracy:** 0.398  
- **Macro F1:** 0.132  
- **Balanced Accuracy:** 0.190 

**Yorum:**  
- **Accuracy (~%40):** Genel doğru tahmin oranı düşük düzeydedir.  
- **Macro F1 (0.13):** Sınıflar arası dengesizliği yansıtarak modelin bazı sınıflarda çok zayıf performans gösterdiğini ortaya koymaktadır.  
- **Balanced Accuracy (~%19):** Tüm sınıfları eşit önemde kabul eden bu metrik de düşük çıkmıştır; bu da modelin genelleme gücünün sınırlı olduğunu göstermektedir.  

Genel tabloya bakıldığında, model doğrulama kümesinde sınıfları ayırt etmede yeterince başarılı değildir ve özellikle dengesiz sınıflarda performans kaybı belirgindir.

---


```python
from sklearn import metrics as M
import numpy as np

y_prob = cnn.predict(val_gen, verbose=0)
y_pred = y_prob.argmax(1)
y_true = val_gen.classes

print("CNN | Accuracy     :", M.accuracy_score(y_true, y_pred))
print("CNN | Macro F1     :", M.f1_score(y_true, y_pred, average='macro'))
print("CNN | Balanced Acc :", M.balanced_accuracy_score(y_true, y_pred))
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
CNN | Accuracy     : 0.39874664279319605
CNN | Macro F1     : 0.13212831645115236
CNN | Balanced Acc : 0.19045190387552957
```


</details>

---


## Confusion Matrix Analizi

**Amaç:**
CNN modelinin sınıflar bazında ne kadar doğru tahmin yaptığını ve hangi sınıfların birbirine karıştığını incelemek.

**Sonuç:**
**Doğru sınıflandırma oranları (diagonal değerler):**
- NV: %78 ile en yüksek doğruluk sağlanan sınıf.
- MEL: %80 doğruluk ile ikinci sırada.
- DF: %48, SCC: %52, VASC: %57 oranında doğru sınıflandırma yapılmış.
- AK: %32, BCC: %0, BKL: %2 gibi düşük doğruluk değerleri gözlenmiş.

**Yanlış sınıflandırmalar:**
- BCC sınıfı hiç doğru tahmin edilememiş, sıklıkla DF, MEL, NV ve VASC sınıflarına karışmış.
- BKL örnekleri %41 oranında MEL sınıfına kaymış.
- AK örnekleri çoğunlukla DF, MEL ve SCC sınıflarıyla karıştırılmış.
- VASC sınıfı %57 oranında doğru yakalansa da sık sık NV ile karışmış.

**Genel tablo:**
Model NV ve MEL sınıflarında güçlü bir ayrıştırma kabiliyetine sahip.BCC, BKL ve AK gibi bazı sınıflarda model başarısız olmuş.Confusion matrix, sınıflar arasında ciddi dengesizlikler ve çakışmalar olduğunu gösteriyor.


---


```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

classes = list(val_gen.class_indices.keys())
cm = confusion_matrix(y_true, y_pred, normalize='true')

# CM ısı haritası
plt.figure(figsize=(7,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix — CNN (class_weight)")
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, f"{cm[i,j]:.2f}", ha='center', va='center', color='black')
plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
plt.tight_layout(); plt.show()

# Sınıf bazlı precision / recall / F1
print(classification_report(y_true, y_pred, target_names=classes, digits=3))
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
<Figure size 700x500 with 2 Axes>
              precision    recall  f1-score   support

          AK      0.210     0.316     0.253       316
         BCC      0.000     0.000     0.000       831
         BKL      0.149     0.024     0.042       573
          DF      0.014     0.188     0.026        48
         MEL      0.258     0.018     0.034       943
          NV      0.599     0.781     0.678      2659
         SCC      0.000     0.000     0.000       164
        VASC      0.013     0.196     0.024        51

    accuracy                          0.399      5585
   macro avg      0.155     0.190     0.132      5585
weighted avg      0.356     0.399     0.348      5585
```

<img width="582" height="460" alt="image" src="https://github.com/user-attachments/assets/2f8b857f-99ee-42ae-ad68-718ddcd5ee47" />


</details>

---


## Classification Report Analizi

**Amaç:**
Modelin doğrulama setindeki her sınıfta ne kadar başarılı olduğunu görmek. Bunun için her sınıf için Precision (kesinlik), Recall (duyarlılık) ve F1 skorları incelenmiştir.

**Precision (Kesinlik):** Modelin bir sınıf için yaptığı tahminlerin ne kadarının doğru olduğunu gösterir.
 “Model X dediğinde, gerçekten X olma ihtimali nedir?”

**Recall (Duyarlılık):** Gerçekten o sınıfa ait örneklerin ne kadarının doğru bulunduğunu gösterir.
 “Tüm X örneklerinin ne kadarını model bulabildi?”

**F1-Score:** Precision ve Recall’un dengeli ortalamasıdır.
“Model hem doğru tahmin yapabiliyor mu, hem de olabildiğince çok doğruyu yakalayabiliyor mu?”

**Sonuçlar:**
- NV (Nevus): En yüksek performans gösteren sınıf. F1 skoru 0.68 ile diğer sınıflardan belirgin şekilde ayrılıyor.
- MEL (Melanoma): Recall çok yüksek (0.80), ancak Precision düşük (0.26). Model MEL örneklerinin çoğunu bulabilmiş ama yanlış pozitifler fazla.
- DF (Dermatofibroma): Recall orta seviyede (0.48) fakat Precision çok düşük (0.01). Model DF örneklerini kısmen yakalamış, ama yanlış sınıflandırma oranı yüksek.
- VASC (Vascular lesions): Recall orta seviyede (0.57), Precision ise çok düşük (0.01). Model birçok örneği VASC’a atamış ama doğru olan az.
- AK, BCC, BKL, SCC: Hem Precision hem Recall çok düşük. Özellikle BCC ve SCC için F1 skoru 0.00’a yakın.

**Genel Tablo:**

- Model büyük sınıf olan NV’de belirgin bir başarı sağlamış.
- MEL sınıfında kısmi bir başarı mevcut, ancak Precision düşük olduğu için güvenilir değil.
- Küçük veya dengesiz sınıflarda modelin performansı çok düşük, çoğu sınıf neredeyse ayırt edilemiyor.
- Genel doğruluk (Accuracy) %39, ortalama F1 skorları da oldukça düşük.

**Yorum:**
Model yalnızca bazı sınıflarda (özellikle NV) anlamlı performans gösterebilmiş. Diğer sınıflarda Precision ve Recall değerleri çok düşük olduğundan, güvenilir bir sınıflandırma yapamıyor. Bu sonuçlar, sınıf dengesizliği ve modelin sınıf sınırlarını yeterince öğrenememesinden kaynaklanıyor.



---


```python
from sklearn.metrics import classification_report

# Sınıf isimlerini generator'dan alıyoruz
classes = list(val_gen.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=classes, digits=3))
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
precision    recall  f1-score   support

          AK      0.210     0.316     0.253       316
         BCC      0.000     0.000     0.000       831
         BKL      0.149     0.024     0.042       573
          DF      0.014     0.188     0.026        48
         MEL      0.258     0.018     0.034       943
          NV      0.599     0.781     0.678      2659
         SCC      0.000     0.000     0.000       164
        VASC      0.013     0.196     0.024        51

    accuracy                          0.399      5585
   macro avg      0.155     0.190     0.132      5585
weighted avg      0.356     0.399     0.348      5585
```


</details>

---


İlk denemede basit bir CNN modeli kullanılmış, ancak modelde **overfitting** görülmüş ve doğruluk yalnızca **%39** seviyesinde kalmıştır. Bu sonuç, daha güçlü ve genelleme kapasitesi yüksek yaklaşımlara ihtiyaç olduğunu göstermektedir.


---


## Model İyileştirmesi — EfficientNetB0 Transfer Learning

**Amaç:**

Basit CNN mimarisinin sınırlı performansını aşmak için, ImageNet üzerinde öncedeneğitilmiş EfficientNetB0 gövdesi kullanılmaktadır. Bu sayede düşük seviyeli 
özelliklerin güçlü temsilleri hazır alınarak,sınıflandırma katmanlarının yeniden eğitilmesiyle genelleme kabiliyeti artırılacaktır.  

**Yöntem:**  
- **Model Gövdesi:** EfficientNetB0, `include_top=False` ve `pooling='avg' kullanılmıştır.  
- **Transfer Learning:** Önceden eğitilmiş taban katmanlarına **freeze** uygulanmış ve yalnızca üstte eklenen sınıflandırıcı katmanlar eğitilmiştir.  
- **Dengesizlik Yönetimi:** `class_weight` parametresi korunarak az temsil edilen sınıfların etkisi güçlendirilmiştir.  
- **Eğitim Süreci:** EarlyStopping, ReduceLROnPlateau ve ModelCheckpoint gibi callback’ler kullanılarak en iyi doğrulama performansı veren model kaydedilmiştir.  

Bu aşama, projenin “Baseline CNN” yaklaşımından sonraki ilk sistematik iyileştirme adımıdır.

**En iyi epoch:** 3

**Val_accuracy:** %48.3

**Val_loss**: 1.477

**Sonuç:**
- Model kısa sürede hızlı bir şekilde öğrenmiş, fakat 3. epoch sonrası overfitting eğilimine girmiş.
- Learning rate azaltılması doğrulama kaybını iyileştirmemiş; dolayısıyla düşük LR ile daha uzun eğitim yapılması da fayda sağlamamış.
- Eğitim doğruluğu sürekli yükselirken doğrulama doğruluğunun durağanlaşması, modelin genel performansının sınırlı kaldığını gösteriyor.

---

```python
# MobileNetV2 tabanlı transfer learning
from tensorflow import keras 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Model gövdesi
base = MobileNetV2(include_top=False, weights="imagenet",
                   input_shape=train_gen.image_shape, pooling="avg")
base.trainable = False   # önce sadece sınıflandırıcı eğitilecek

# Output layer (Classifier layer)
x = layers.Dropout(0.3)(base.output)
out = layers.Dense(train_gen.num_classes, activation="softmax")(x)

mobilenet_model = models.Model(inputs=base.input, outputs=out)

mobilenet_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callback’ler 
early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
ckpt  = ModelCheckpoint("/kaggle/working/mobilenetv2_best.keras",
                        monitor="val_loss", save_best_only=True, verbose=1)

# Eğitim
EPOCHS = 5
history_mnet = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early, rlrop, ckpt],
    class_weight=class_weight,
    verbose=1
)

# Fine-tuning son katmanları açma
for layer in base.layers[-10:]:
    layer.trainable = True

mobilenet_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_mnet_ft = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3,
    callbacks=[early, rlrop, ckpt],
    class_weight=class_weight,
    verbose=1
)
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
Epoch 1/5
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410ms/step - accuracy: 0.3562 - loss: 2.0786
Epoch 1: val_loss improved from inf to 1.87552, saving model to /kaggle/working/mobilenetv2_best.keras
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m319s[0m 442ms/step - accuracy: 0.3563 - loss: 2.0782 - val_accuracy: 0.3284 - val_loss: 1.8755 - learning_rate: 0.0010
Epoch 2/5
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417ms/step - accuracy: 0.4564 - loss: 1.5377
Epoch 2: val_loss improved from 1.87552 to 1.58373, saving model to /kaggle/working/mobilenetv2_best.keras
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m306s[0m 438ms/step - accuracy: 0.4565 - loss: 1.5377 - val_accuracy: 0.4317 - val_loss: 1.5837 - learning_rate: 0.0010
Epoch 3/5
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392ms/step - accuracy: 0.4799 - loss: 1.4076
Epoch 3: val_loss improved from 1.58373 to 1.47723, saving model to /kaggle/working/mobilenetv2_best.keras
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m287s[0m 410ms/step - accuracy: 0.4799 - loss: 1.4076 - val_accuracy: 0.4834 - val_loss: 1.4772 - learning_rate: 0.0010
Epoch 4/5
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393ms/step - accuracy: 0.5032 - loss: 1.3356
Epoch 4: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.

Epoch 4: val_loss did not improve from 1.47723
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m289s[0m 413ms/step - accuracy: 0.5032 - loss: 1.3356 - val_accuracy: 0.4587 - val_loss: 1.5450 - learning_rate: 0.0010
Epoch 5/5
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405ms/step - accuracy: 0.5196 - loss: 1.2933
Epoch 5: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.

Epoch 5: val_loss did not improve from 1.47723
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m300s[0m 429ms/step - accuracy: 0.5196 - loss: 1.2933 - val_accuracy: 0.4346 - val_loss: 1.5867 - learning_rate: 5.0000e-04
Restoring model weights from the end of the best epoch: 3.
Epoch 1/3
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406ms/step - accuracy: 0.4746 - loss: 1.6284
Epoch 1: val_loss did not improve from 1.47723
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m320s[0m 437ms/step - accuracy: 0.4747 - loss: 1.6282 - val_accuracy: 0.4571 - val_loss: 2.0048 - learning_rate: 1.0000e-04
Epoch 2/3
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402ms/step - accuracy: 0.5180 - loss: 1.2561
Epoch 2: val_loss did not improve from 1.47723
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m298s[0m 426ms/step - accuracy: 0.5180 - loss: 1.2561 - val_accuracy: 0.5046 - val_loss: 1.7826 - learning_rate: 1.0000e-04
Epoch 3/3
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406ms/step - accuracy: 0.5408 - loss: 1.1590
Epoch 3: val_loss did not improve from 1.47723
[1m699/699[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m298s[0m 426ms/step - accuracy: 0.5408 - loss: 1.1589 - val_accuracy: 0.4992 - val_loss: 1.5561 - learning_rate: 1.0000e-04
Restoring model weights from the end of the best epoch: 3.
```


</details>

---


## MobileNetV2 Eğitim Sonuçları

**Amaç:**
Basit CNN modelinde yaşanan overfitting’i azaltmak ve doğrulama setinde daha dengeli bir performans elde etmek için MobileNetV2 tabanlı transfer learning uygulandı.

**Sonuç:**
- Eğitim doğruluğu %39’dan %54’e yükseldi.
- Doğrulama doğruluğu başlangıçta %32 seviyesindeydi, eğitim süresince %48–%51 aralığında dalgalandı.
- Eğitim ve doğrulama doğruluk eğrileri genel olarak paralel ilerledi, sadece 5. epoch civarında geçici bir düşüş görüldü.
- Eğitim kaybı düzenli şekilde azaldı (1.9 → 1.1), doğrulama kaybı ise dalgalı bir seyir izledi; 6. epoch’ta 2.0 seviyesine çıksa da daha sonra tekrar 1.5 seviyelerine geriledi.

**Önceki CNN denemesine göre fark:**
- CNN’de model eğitim verisine aşırı uyum sağlamış ve doğrulama performansı hızla bozulmuştu (belirgin overfitting).
- MobileNetV2’de bu sorun belirgin şekilde azaldı; model train ve val setlerinde daha dengeli ilerledi.
- Genel doğruluk seviyeleri de CNN’e göre daha yüksek oldu (yaklaşık %48 → %51 val doğruluğu).

 **Özet:** MobileNetV2, CNN’e kıyasla daha kararlı ve dengeli bir öğrenme süreci sağlamış, özellikle doğrulama setinde performans kaybı daha sınırlı kalmıştır.


---


```python
import matplotlib.pyplot as plt

# history_mnet (feature extraction) ve history_mnet_ft (fine-tuning) 
#varsa birleştiriyoruz
acc = history_mnet.history['accuracy'] + history_mnet_ft.history['accuracy']
val_acc = history_mnet.history['val_accuracy'] + history_mnet_ft.history['val_accuracy']
loss = history_mnet.history['loss'] + history_mnet_ft.history['loss']
val_loss = history_mnet.history['val_loss'] + history_mnet_ft.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10,4))

# Accuracy grafiği
plt.subplot(1,2,1)
plt.plot(epochs, acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.title('MobileNetV2 | Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend()

# Loss grafiği
plt.subplot(1,2,2)
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('MobileNetV2 | Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
<Figure size 1000x400 with 2 Axes>
```

<img width="1028" height="399" alt="image" src="https://github.com/user-attachments/assets/992fc7be-89c2-4883-831f-95b2635c3ec2" />


</details>

---


## CNN vs MobileNetV2 — Sonuç Karşılaştırması
Amaç: İyileştirmeyi somut göstermek için her iki modelin doğrulama sonuçları yan yana sunulmuştur.


---


```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as M

y_true = val_gen.classes

# CNN sonuçları
y_pred_cnn = cnn.predict(val_gen, verbose=0).argmax(1)
cnn_acc  = M.accuracy_score(y_true, y_pred_cnn)
cnn_f1   = M.f1_score(y_true, y_pred_cnn, average='macro')
cnn_bacc = M.balanced_accuracy_score(y_true, y_pred_cnn)

# MobileNetV2 sonuçları
y_pred_mnv2 = mobilenet_model.predict(val_gen, verbose=0).argmax(1)
mnv2_acc  = M.accuracy_score(y_true, y_pred_mnv2)
mnv2_f1   = M.f1_score(y_true, y_pred_mnv2, average='macro')
mnv2_bacc = M.balanced_accuracy_score(y_true, y_pred_mnv2)

# Tablo oluştur
df_cmp = pd.DataFrame({
    "Model": ["CNN (Baseline)", "MobileNetV2 (TL+FT)"],
    "Accuracy": [cnn_acc, mnv2_acc],
    "Macro F1": [cnn_f1, mnv2_f1],
    "Balanced Acc": [cnn_bacc, mnv2_bacc]
}).round(3)

display(df_cmp)

# Grafik
df_plot = df_cmp.set_index("Model")
df_plot.plot(kind="bar", figsize=(8,5))
plt.title("CNN vs MobileNetV2 — Sonuç Karşılaştırması")
plt.ylabel("Skor")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(title="Metrik")
plt.show()
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
Model  Accuracy  Macro F1  Balanced Acc
0       CNN (Baseline)     0.399     0.132         0.190
1  MobileNetV2 (TL+FT)     0.499     0.280         0.373
<Figure size 800x500 with 1 Axes>
```

<img width="738" height="466" alt="image" src="https://github.com/user-attachments/assets/e2c97994-8444-4947-a22b-789714c2bcdc" />


</details>

---


## Grad-CAM Değerlendirmesi

**Amaç:**  
Modelin karar verirken lezyon bölgesine odaklanıp odaklanmadığını görmek.  

**Sonuç:**  
- Doğru tahminlerde ısı haritası genellikle lezyonun merkezinde yoğunlaştı.  
- Yanlış tahminlerde odak dağıldı veya lezyon dışına kaydı.  
- Özellikle AK–BKL–SCC gibi benzer sınıflarda karışmalar bu yüzden arttı.  

**Yorum:**  
Model çoğunlukla doğru bölgeleri kullanıyor; ancak benzer sınıflarda odak kaymaları hatalı sınıflandırmalara yol açıyor.


---

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_last_conv_layer(model):
    """Modeldeki son (batch, H, W, C) çıkışlı katmanı bulur."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("Uygun bir konvolüsyon katmanı bulunamadı.")

def gradcam_heatmap(model, img_tensor, conv_layer_name, pred_index=None):
    """
    Tek bir görüntü (1,H,W,3) için Grad-CAM ısı haritası döndürür: (Hh,Wh) [0..1].
    """
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], 
                                       [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_out)           # d(class)/d(featuremap)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))           # kanal bazında ortalama
    conv_out = conv_out[0]                                  # (Hh,Wh,C)
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1).numpy() # (Hh,Wh)

    heatmap = np.maximum(heatmap, 0)
    m = heatmap.max()
    return heatmap / (m + 1e-12)

def show_gradcam_grid(model, val_gen, n=6):
    """
    Val setinden 6 örnek seçer (3 doğru + 3 yanlış; yoksa rastgele tamamlar),
    Grad-CAM ısı haritalarını orijinal görüntünün üzerine bindirerek 
    2x3 gridde gösterir.
    """
    classes = list(val_gen.class_indices.keys())
    y_true = val_gen.classes
    bsz = val_gen.batch_size

    # Tahminleri al ve indeksleri hazırla
    preds_all = model.predict(val_gen, verbose=0).argmax(1)
    idx_correct = np.where(preds_all == y_true)[0].tolist()
    idx_wrong   = np.where(preds_all != y_true)[0].tolist()

    pick = (idx_correct[: n//2] + idx_wrong[: n - n//2])[:n]
    if len(pick) < n:
        pool = np.arange(len(y_true))
        np.random.shuffle(pool)
        for i in pool:
            if i not in pick:
                pick.append(i)
            if len(pick) == n:
                break

    # Son konv katmanı (otomatik)
    conv_name = get_last_conv_layer(model)
    print("Son konvolüsyon katmanı:", conv_name)
 

     # Çizim
    plt.figure(figsize=(12, 8))
    for k, gi in enumerate(pick, start=1):
        bi, ii = gi // bsz, gi % bsz
        x_batch, y_batch = val_gen[bi]
        img = x_batch[ii]                                 # [H,W,3], 0-1
        true_id = int(y_batch[ii])
        img_in = np.expand_dims(img, axis=0).astype("float32")

        hm = gradcam_heatmap(model, img_in, conv_name)
        # TF ile orijinal boyuta ölçekle
        hm_tf = tf.expand_dims(tf.expand_dims(hm, 0), -1) # [1,Hh,Wh,1]
        hm_rs = tf.image.resize(hm_tf, (img.shape[0], img.shape[1]))
        hm_rs = tf.squeeze(hm_rs, [0, -1]).numpy()

        plt.subplot(2, 3, k)
        plt.imshow(img)                                     # orijinal
        plt.imshow(hm_rs, cmap="jet", alpha=0.35)         # ısı haritası
        pred_id = int(preds_all[gi])
        plt.title(f"T:{classes[true_id]} | P:{classes[pred_id]}")
        plt.axis("off")

    plt.suptitle("Grad-CAM — MobileNetV2 (basit sürüm)")
    plt.tight_layout()
    plt.show()
     # === KULLANIM ===
     show_gradcam_grid(mobilenet_model, val_gen, n=6)



```

<img width="1153" height="788" alt="download" src="https://github.com/user-attachments/assets/1a2f7c23-8d20-475c-9935-9fb78540752f" />
</details>

```

---
```


 ## Confusion Matrix (MobileNetV2)

**Amaç:**
Hangi sınıfların doğru/yanlış sınıflandığını ve en çok hangi sınıflarla karıştığını görmek.

**Gözlemler:**

- **VASC** sınıfı **%90** ile en yüksek doğruluk oranına sahiptir. Model bu sınıfı çok güçlü şekilde ayırt edebilmiştir.
- **NV** sınıfı **%76** ile ikinci sırada yer almakta, en büyük sınıf olmasına rağmen iyi bir doğruluk sergilemektedir.
- **BKL** **%46**, **MEL** **%46** ve **DF** **%50** oranlarında orta düzeyde doğru sınıflandırılmıştır.
- **BCC** **%25** ve **AK** **%9** oranında doğru sınıflandırma ile düşük başarı göstermiştir.
- **SCC** ***%41** oranında kısmi başarı sağlamış olsa da karışma oranı yüksektir.
- Özellikle **AK** → **BKL**, **SCC**→**BKL**, **MEL** → **BKL** yönlü yanlış sınıflandırmalar dikkat çekmektedir.

**Sonuç:**
MobileNetV2 modeli, **VASC** ve **NV** gibi sınıflarda güçlü bir performans göstermiş, **BKL**, **MEL**, **DF** gibi sınıflarda orta düzeyde başarı sağlamıştır. Ancak AK, BCC gibi küçük veya zor sınıflarda düşük doğruluk elde edilmiştir. Yanlış sınıflandırmalar, genellikle görsel olarak benzer lezyon tiplerinde yoğunlaşmaktadır. Bu tablo, modelin güçlü yönlerini (**VASC**, **NV**) ve zorlandığı alanları (**AK**, **BCC**, **SCC**) net biçimde ortaya koymaktadır.

---

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

classes = list(val_gen.class_indices.keys())
y_true = val_gen.classes
y_pred_mnv2 = mobilenet_model.predict(val_gen, verbose=0).argmax(1)

cm = confusion_matrix(y_true, y_pred_mnv2, normalize='true')

plt.figure(figsize=(7,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix — MobileNetV2 (Normalized)")
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, f"{cm[i,j]:.2f}", ha='center', va='center', color='black')
plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
plt.tight_layout(); plt.show()
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
<Figure size 700x500 with 2 Axes>
```

<img width="568" height="433" alt="image" src="https://github.com/user-attachments/assets/af705ef6-ae8c-44d2-af24-db88511e06a3" />


</details>

---

## Classification Report — MobileNetV2

**Amaç:**
 Her sınıf için Precision, Recall ve F1 skorlarını görmek ve modelin hangi sınıflarda daha başarılı, hangi sınıflarda zorlandığını incelemek.
  
**Precision:** Modelin yaptığı tahminlerin doğruluk oranı.

**Recall:** Gerçek örneklerin ne kadarının doğru bulunduğu.

**F1:** Precision ve Recall’un dengeli ortalaması.

**Genel Başarı:**

Modelin toplam doğruluğu %50 civarındadır. Weighted F1 skoru da %49 ile doğrulukla uyumlu görünmektedir.

**Sınıf Bazlı Gözlemler:**

**NV (Nevus):** En iyi sonuç veren sınıf oldu (F1 ≈ 0.73). Bu sınıfta veri sayısının fazla olması modele avantaj sağlamış.

**BCC (Basal Cell Carcinoma)** ve **MEL (Melanoma)**: Orta düzeyde başarı elde edildi (F1 ≈ 0.31–0.32).

**BKL (Benign Keratosis)**: Recall oldukça yüksek (0.46), ancak Precision düşük (0.23). Model bu sınıfa çok tahmin yapıyor ama çoğu yanlış.

**VASC (Vascular Lesions):** Recall çok yüksek (0.90), Precision düşük (0.11). Model bu sınıfı neredeyse her zaman yakalıyor, ancak sık yanlış pozitif üretiyor.

**AK, DF, SCC:** Skorlar oldukça düşük. Özellikle DF (F1 ≈ 0.13) ve SCC (F1 ≈ 0.08) sınıflarında modelin ayırt edici gücü zayıf kalmış.

Sonuç:
MobileNetV2, NV sınıfında belirgin bir başarı göstermiş, BKL, MEL ve BCC’de orta düzeyde performans sağlamıştır. Ancak AK, DF ve SCC gibi küçük veya görsel olarak benzer sınıflarda model ciddi zorluk yaşamaktadır. VASC sınıfında model örneklerin çoğunu yakalayabilse de düşük Precision sebebiyle güvenilirlik sınırlıdır.

Bu tablo, dengesiz veri dağılımı ve benzer görünümlü sınıflar nedeniyle modelin genelleme kabiliyetinde eksiklik olduğunu ortaya koymaktadır. Daha iyi sonuç için data augmentation, oversampling, class weighting veya focal loss gibi yöntemler kullanılabilir.

---


```python
from sklearn.metrics import classification_report

y_true = val_gen.classes
y_pred = mobilenet_model.predict(val_gen, verbose=0).argmax(1)
classes = list(val_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=classes, digits=3))
```

<details>
<summary><strong>Show cell outputs</strong></summary>

```text
precision    recall  f1-score   support

          AK      0.600     0.085     0.150       316
         BCC      0.581     0.212     0.310       831
         BKL      0.231     0.457     0.307       573
          DF      0.096     0.229     0.135        48
         MEL      0.464     0.243     0.319       943
          NV      0.706     0.760     0.732      2659
         SCC      0.070     0.098     0.082       164
        VASC      0.114     0.902     0.202        51

    accuracy                          0.499      5585
   macro avg      0.358     0.373     0.280      5585
weighted avg      0.562     0.499     0.494      5585
```


</details>

---


## Final Özet

**Amaç:**
Cilt lezyonlarını sınıflandırmak için CNN tabanlı bir model geliştirmek ve farklı yöntemleri deneyerek en iyi sonucu elde etmek.

**Yöntem:**

- Önce basit bir CNN baseline modeli kuruldu. Bu modelde doğrulama doğruluğu düşük kaldı ve belirgin overfitting görüldü.Daha sonra ImageNet üzerinde önceden eğitilmiş MobileNetV2 tabanı kullanılarak transfer learning yapıldı. İlk aşamada yalnızca eklenilen sınıflandırıcı katman eğitildi (feature extraction). Ardından son katmanlar açılarak düşük öğrenme oranıyla **fine-tuning** yapıldı.Eğitim sürecinde **EarlyStopping,** **ReduceLROnPlateau** ve **ModelCheckpoint** callback’leri kullanıldı. Modeller ***Accuracy/Loss** grafikleri, **Classification Report**, **Confusion Matrix** ve **Grad-CAM** ile detaylı şekilde değerlendirildi. Ayrıca küçük bir hiperparametre denemesi yapılarak dropout ve öğrenme oranının etkisi gözlemlendi.
  
**Sonuç:**
CNN baseline düşük başarı gösterdi (Accuracy ≈ %39, Macro F1 ≈ 0.13). MobileNetV2 ile doğruluk %50’ye yükseldi, Macro F1 0.28, Weighted F1 ise 0.49 oldu. Overfitting azaldı, doğrulama performansı daha dengeli hale geldi. Büyük sınıflarda (ör. NV, VASC) yüksek başarı sağlanırken, küçük ve benzer görünümlü sınıflarda (AK, DF, SCC) karışmalar devam etti. Grad-CAM görselleri, doğru sınıflarda modelin lezyon bölgesine odaklandığını, yanlışlarda ise odak kaymaları yaşadığını gösterdi.

**Genel Değerlendirme:**
Transfer learning ile elde edilen MobileNetV2 modeli, basit CNN’e kıyasla belirgin bir iyileşme sağladı. Veri dengesizliği ve benzer sınıfların ayrımı hâlâ zorlayıcı olsa da, proje amacı doğrultusunda başarılı bir temel model elde edildi.


Kaggle Linki: https://www.kaggle.com/code/beyzanurunlu/skin-lesion-classification
### How to Reproduce
---



```bash
pip install -r requirements.txt  # if you have one
jupyter nbconvert --to notebook --execute skin-lesion-classification.ipynb
```

_Generated automatically. Feel free to edit headings and descriptions for clarity._
