import usb.core
import usb.backend.libusb1
import time

# 경광등의 USB 벤더 ID와 프로덕트 ID입니다. 실제 기기의 값으로 변경해주세요.
VENDOR_ID = 0x16c0
PRODUCT_ID = 0x27d9

# 경광등이 깜박이는 주기입니다.
BLINK_INTERVAL = 0.1

# libusb1 backend를 사용합니다.
backend = usb.backend.libusb1.get_backend()

# 경광등을 찾습니다.
device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID, backend=backend)
if device is None:
    raise ValueError("Device not found")

# 경광등이 사용 중인 인터페이스와 엔드포인트를 찾습니다.
interface = 0
endpoint = device[0][(0, 0)][0]

# 경광등을 깜박입니다.
while True:
    endpoint.write([0x01])
    time.sleep(BLINK_INTERVAL)
    endpoint.write([0x00])
    time.sleep(BLINK_INTERVAL)