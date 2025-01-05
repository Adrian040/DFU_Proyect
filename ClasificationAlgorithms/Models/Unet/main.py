import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import  torchvision.transforms.functional as TF


## ------------------ Unet -------------------

# Bloque de código de cada capa de la Unet (donde se aplican las dos convoluciones, Además se aplica el batchnorm):
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_prob=0.5):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# Implement Dropout:
impl_dropout = True
prob_dropout = 0.4

# Clase principal de la Unet:
class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)

        # Down part of Unet:
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, use_dropout=impl_dropout, dropout_prob=prob_dropout))
            in_channels = feature

        # Up part of Unet:
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature, use_dropout=impl_dropout, dropout_prob=prob_dropout))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2, use_dropout=impl_dropout, dropout_prob=prob_dropout) # Medium part
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) #Final part


    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)



def test_model():
    batch_size = 2  # Número de imágenes en el lote
    in_channels = 3  # Canales de entrada (RGB)
    height, width = 224, 224  # Tamaño de las imágenes de entrada
    num_classes = 4  # Número de clases para segmentación

    # Crear un tensor aleatorio para simular las imágenes de entrada
    x = torch.randn(batch_size, in_channels, height, width)

    # Instanciar el modelo
    model = UNET(in_channels=in_channels, out_channels=num_classes)

    # Pasar el tensor de entrada a través del modelo
    outputs = model(x)

    # Imprimir las dimensiones de la salida
    # print(f"Dimensiones de salida: {outputs.shape}")
    # print(f"Esperado: ({batch_size}, {num_classes}, {height}, {width})")

    # Verificar que las dimensiones sean las correctas
    assert outputs.shape == (batch_size, num_classes, height, width), \
        "Error: Las dimensiones de la salida no son correctas."

    print("El modelo funciona correctamente.")

# Llamar a la función de prueba:
print("Probando el modelo...")
start=time.time()
test_model()
print("(Time of 2 224x224 imgs: ",round(time.time()-start, 2), "s)\n\n")