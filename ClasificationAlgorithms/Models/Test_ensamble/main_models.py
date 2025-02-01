import torch
import torch.nn as nn
import torch.nn.functional as F
import  torchvision.transforms.functional as TF

# -------------------- Attention U-Net --------------------------------

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn_relu = batchnorm_relu(out_c)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn_relu(x)
        x = self.dropout(x)
        return x
    
class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsample para g

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Aplicar upsampling a g1 para que coincida con las dimensiones de x1
        g1 = self.upsample(g1)
        
        psi = F.relu(g1 + x1)  # Sumar g1 y x1
        psi = self.psi(psi)
        return x * psi


class AttentionUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout=0.5):
        super(AttentionUnet, self).__init__()
        self.enc1 = conv_block(in_channels, 64, dropout)
        self.enc2 = conv_block(64, 128, dropout)
        self.enc3 = conv_block(128, 256, dropout)
        self.enc4 = conv_block(256, 512, dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = conv_block(512, 1024, dropout)

        # Ajustar F_g para que coincida con el número de canales de center (1024)
        self.att4 = attention_block(F_g=1024, F_l=512, F_int=512)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512, dropout)

        self.att3 = attention_block(F_g=512, F_l=256, F_int=256)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256, dropout)

        self.att2 = attention_block(F_g=256, F_l=128, F_int=128)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128, dropout)

        self.att1 = attention_block(F_g=128, F_l=64, F_int=64)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64, dropout)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, inputs):
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        att4 = self.att4(g=center, x=enc4)
        up4 = self.up4(center)
        merge4 = torch.cat([up4, att4], dim=1)
        dec4 = self.dec4(merge4)

        att3 = self.att3(g=dec4, x=enc3)
        up3 = self.up3(dec4)
        merge3 = torch.cat([up3, att3], dim=1)
        dec3 = self.dec3(merge3)

        att2 = self.att2(g=dec3, x=enc2)
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, att2], dim=1)
        dec2 = self.dec2(merge2)

        att1 = self.att1(g=dec2, x=enc1)
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, att1], dim=1)
        dec1 = self.dec1(merge1)

        output = self.final_conv(dec1)
        return output
    

# ---------------------------------------------------------------

## ------------------------ ResUnet --------------------------------
class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=0.5):
        super().__init__()
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.dropout1 = nn.Dropout2d(p=dropout)  # Dropout agregado
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.dropout2 = nn.Dropout2d(p=dropout)  # Dropout agregado
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.dropout1(x)  # Aplicar Dropout después de la primera convolución
        x = self.b2(x)
        x = self.c2(x)
        x = self.dropout2(x)  # Aplicar Dropout después de la segunda convolución
        s = self.s(inputs)
        skip = x + s
        return skip

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.5):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dropout = nn.Dropout2d(p=dropout)  # Dropout antes de la convolución
        self.r = residual_block(in_c + out_c, out_c, dropout=dropout)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.dropout(x)  # Aplicar Dropout después de la concatenación
        x = self.r(x)
        return x

class ResUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout=0.5):
        super().__init__()
        self.c11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)

        self.r2 = residual_block(64, 128, stride=2, dropout=dropout)
        self.r3 = residual_block(128, 256, stride=2, dropout=dropout)
        self.r4 = residual_block(256, 512, stride=2, dropout=dropout)

        self.d1 = decoder_block(512, 256, dropout=dropout)
        self.d2 = decoder_block(256, 128, dropout=dropout)
        self.d3 = decoder_block(128, 64, dropout=dropout)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, inputs):
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        b = self.r4(skip3)

        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        output = self.final_conv(d3)
        return output
    
# ---------------------------------------------------------------.

## ------------------------ Unet --------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_prob > 0.0 and dropout_prob < 1.0:
            layers.append(nn.Dropout(dropout_prob))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
        dropout_prob=0.0
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)

        # Down part of Unet:
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_prob=dropout_prob))
            in_channels = feature

        # Up part of Unet:
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature, dropout_prob=dropout_prob))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_prob=dropout_prob) # Medium part
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
    

# ---------------------------------------------------------------.

## ------------------------ Unet++ --------------------------------

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn_relu = batchnorm_relu(out_c)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn_relu(x)
        x = self.dropout(x)
        return x

class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout=0.5):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (X_{0,0}, X_{1,0}, X_{2,0}, X_{3,0}, X_{4,0})
        self.enc1 = conv_block(in_channels, 64, dropout)
        self.enc2 = conv_block(64, 128, dropout)
        self.enc3 = conv_block(128, 256, dropout)
        self.enc4 = conv_block(256, 512, dropout)
        self.enc5 = conv_block(512, 1024, dropout)  # Centro (X_{4,0})

        # Intermedios (X_{i,j})
        self.x01 = conv_block(64 + 64, 64, dropout)  # Ajustado para 64 + 64
        self.x11 = conv_block(128 + 128, 128, dropout)
        self.x21 = conv_block(256 + 256, 256, dropout)
        self.x31 = conv_block(512 + 512, 512, dropout)

        self.x02 = conv_block(64 * 2 + 64, 64, dropout)
        self.x12 = conv_block(128 * 2 + 128, 128, dropout)
        self.x22 = conv_block(256 * 2 + 256, 256, dropout)

        self.x03 = conv_block(64 * 3 + 64, 64, dropout)
        self.x13 = conv_block(128 * 3 + 128, 128, dropout)

        self.x04 = conv_block(64 * 4 + 64, 64, dropout)

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Ajustado para 128 -> 64
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        # Salida final
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, inputs):
        # Encoder
        x00 = self.enc1(inputs)
        x10 = self.enc2(self.pool(x00))
        x20 = self.enc3(self.pool(x10))
        x30 = self.enc4(self.pool(x20))
        x40 = self.enc5(self.pool(x30))

        # Decodificación anidada
        x01 = self.x01(torch.cat([x00, self.up1(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up2(x20)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up3(x30)], dim=1))
        x31 = self.x31(torch.cat([x30, self.up4(x40)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self.up1(x11)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up2(x21)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self.up3(x31)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up1(x12)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up2(x22)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up1(x13)], dim=1))

        # Salida
        output = self.final_conv(x04)
        return output
    

def test():
    x = torch.randn((3, 1, 160, 160))  # Dimensiones deben ser pares
    model = ResUnet(in_channels=1, out_channels=1, dropout=0.5)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape

test()
