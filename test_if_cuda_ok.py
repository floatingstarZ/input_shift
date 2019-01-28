import torch
def test_gpu():
    if torch.cuda.is_available():
        print('cuda acailable')
    else:
        print('cuda not acailable')
    devices = torch.cuda.device_count()
    print('There are %d gpu devices.' %
          devices)
    for d in range(devices):
        print('Device %d : %s' % (
            d, torch.cuda.get_device_name(d)))
    print('Current Device: %s'%
          torch.cuda.current_device())

