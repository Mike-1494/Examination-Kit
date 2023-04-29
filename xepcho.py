import random
import numpy as np

num_seats = int(input("Nhap so cho ngoi trong phong thi: "))
seat = np.zeros(num_seats, dtype=bool)

def seat_taker(num_seats: int) -> int:
    while True:
        x = random.randint(0, num_seats-1)
        if not seat[x]:
            seat[x] = True
            return x
if(__name__ == "__main__"):
    print(seat_taker(num_seats))
