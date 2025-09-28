import qrcode

data = "user1"
img = qrcode.make(data)
img.save("image/qrcode_user1.png")