import serial
import time
import serial.tools.list_ports
for item in serial.tools.list_ports.comports():
  print( item )
ser = serial.Serial('/dev/ttyACM0', 115200, 8, 'N', 1, timeout=1)
# initialize the odometer values
mm_per_tick = 0.0000599

ser.write(str.encode('reset'))  # restart the encoder to zero

def update():
    last_time = time.time()
    ticks = 0
    meters = 0
    meters_per_second = 0
    # keep looping infinitely until the thread is stopped
    while True:
        input = ser.readline()
        ticks = input.decode()
        print("ticks=", ticks)
        ticks = ticks.strip()  # remove any whitespace
        if ticks.isnumeric():
            ticks = int(ticks)
            if ticks > 0: 
                #save off the last time interval and reset the timer
                start_time = last_time
                end_time = time.time()
                last_time = end_time
                
                #calculate elapsed time and distance traveled
                seconds = end_time - start_time
                distance = ticks * mm_per_tick
                velocity = distance / seconds

        #       print('seconds:', seconds)
                print('distance (m):', distance)
                print('velocity (m/s):', velocity)


update()