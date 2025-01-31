import torch
import torch.nn as nn

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
        self.x01 = conv_block(64 + 128, 64, dropout)
        self.x11 = conv_block(128 + 256, 128, dropout)
        self.x21 = conv_block(256 + 512, 256, dropout)
        self.x31 = conv_block(512 + 1024, 512, dropout)

        self.x02 = conv_block(64 * 2 + 128, 64, dropout)
        self.x12 = conv_block(128 * 2 + 256, 128, dropout)
        self.x22 = conv_block(256 * 2 + 512, 256, dropout)

        self.x03 = conv_block(64 * 3 + 128, 64, dropout)
        self.x13 = conv_block(128 * 3 + 256, 128, dropout)

        self.x04 = conv_block(64 * 4 + 128, 64, dropout)

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
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
    model = UnetPlusPlus(in_channels=1, out_channels=1, dropout=0.5)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape  # Esto fallaría si no se restaura la resolución espacial

test()
