import tensorflow as tf 
import argparse

def activate_GPU():
    gpus = tf.config.list_physical_devices('GPU')
    print("="*50)
    if len(gpus):
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Run code with GPU:0")
    else:
        print("Run code with CPU")
    print("="*50)
    
def parse_args():
    # Tạo một đối tượng ArgumentParser
    parser = argparse.ArgumentParser()

    # Thêm các đối số dòng lệnh
    parser.add_argument("--max_step", default=500, type=int)
    parser.add_argument("--episode", default=1000, type=int)

    # Phân tích các đối số từ dòng lệnh
    args = parser.parse_args()
    return args

def write_file(file_name, data):
    # Open the file in append mode
    with open(file_name, 'a') as file:
        # Write the data to the file
        file.write(str(data))
        file.write('\n')
