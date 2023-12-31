# file for python alarm clock
import check_water
import subprocess
from pygame import mixer
from tkinter import *
from threading import *
import datetime
import time
import sys

# Global GUI variables
root = Tk()
set_alarm_button = None
snooze_button = None
stop_alarm_button = None
hour = StringVar(root)
minute = StringVar(root)

# Global alarm variables
snoozes_left = 3
volume_lock = True
mixer.init()
time_to_wake_up = False 
alarm_thread = None
alarm_stop_event = Event()
music_playing = False

# alarm function
def alarm():
    global time_to_wake_up
    alarm_time = f"{hour.get()}:{minute.get()}:00"
    while alarm_stop_event.is_set() == False:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        if now == alarm_time:
            # change alarm to wake up mode
            print("Wake up!")
            time_to_wake_up = True
            # lock volume to max
            volume_lock_thread = Thread(target=lock_system_volume, daemon=True)
            volume_lock_thread.start()
            # lock ability to set alarm and play sound
            lock_alarm()
            play_sound()
        time.sleep(1)  # Wait for 1 second before checking the time again
        print(f'Current time: {now} Alarm time: {alarm_time}')

# start alarm function with threading
def start_alarm():
    global alarm_thread
    # stop current alarm (if any)
    if alarm_thread and alarm_thread.is_alive():
        stop_alarm()

    # reset stop event
    alarm_stop_event.clear()

    # start new alarm
    alarm_thread = Thread(target=alarm)
    alarm_thread.start()

# stop current alarm and music
def stop_alarm():
    global alarm_thread, time_to_wake_up, stop_alarm_button, volume_lock
    if alarm_thread == None:
        return
    
    # check if its time to wake up (alarm went off)
    if time_to_wake_up:
        # check if water is running
        stop_alarm_button['state'] = DISABLED
        
        # stop sound and check if water is running
        stop_sound()
        running = check_water.check()
        play_sound()

        # handle cases
        if running:
            # stop alarm and reset alarm to normal
            print("Alarm deactivated! Good morning!")
            volume_lock = False
            stop_sound()
            alarm_stop_event.set()
            alarm_thread.join()
            alarm_thread = None
            sys.exit()
        else:
            print("Water is not running! Alarm still active!")
            stop_alarm_button['state'] = NORMAL
            return

    # stop alarm (if not time to wake up)
    alarm_stop_event.set()
    alarm_thread.join()
    alarm_thread = None

# plays the music
def play_sound():
    global music_playing
    if music_playing:
        return
    # loop until stopped
    mixer.music.load('Audio/FitnessGram.mp3')
    mixer.music.play(loops=-1)
    music_playing = True

# stops the music
def stop_sound():
    global music_playing
    if not music_playing:
        return
    mixer.music.stop()
    music_playing = False

def set_system_volume_to_max():
    subprocess.call(["osascript", "-e", "set volume output volume 100"])

def lock_system_volume():
    global volume_lock
    while volume_lock:
        set_system_volume_to_max()
        time.sleep(1)

# Functions to handle the snooze button press
def snooze_alarm():
    time.sleep(5) # set to however long you want the snooze to be
    if alarm_thread == None:
        return
    play_sound()

# snoozes alarm
def snooze():
    global alarm_thread, snoozes_left
    # check if alarm is active
    if alarm_thread == None:
        return
    # handle repeated snooze button presses
    if snoozes_left == 0:
        return
    if snoozes_left == 1:
        lock_snooze()
    snoozes_left -= 1

    # stop music and start snooze thread
    if alarm_thread == None:
        return
    if music_playing:
        stop_sound()
    snooze_thread = Thread(target=snooze_alarm, daemon=True)
    snooze_thread.start()

# Lock set alarm button
def lock_alarm():
    global set_alarm_button
    if set_alarm_button['state'] == NORMAL:
        set_alarm_button['state'] = DISABLED

# Lock snooze button
def lock_snooze():
    global snooze_button
    if snooze_button['state'] == NORMAL:
        snooze_button['state'] = DISABLED

# Function to disable the close button
def disable_event():
    pass

# GUI initialization function
def init_gui():
    global set_alarm_button, snooze_button, stop_alarm_button, hour, minute, root
    root.geometry("400x400")
    root.title("Alarm Clock")
    root.protocol("WM_DELETE_WINDOW", disable_event)

    # Add Labels, Frame, Button, Optionmenus
    Label(root,text="Alarm Clock",font=("Helvetica 20 bold"),fg="red").pack(pady=10)
    Label(root,text="Set Time",font=("Helvetica 15 bold")).pack()
    
    frame = Frame(root)
    frame.pack()
    
    hours = ('00', '01', '02', '03', '04', '05', '06', '07',
            '08', '09', '10', '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', '21', '22', '23',
            )
    hour.set(hours[0])
    
    hrs = OptionMenu(frame, hour, *hours)
    hrs.pack(side=LEFT)
    
    minutes = ('00', '01', '02', '03', '04', '05', '06', '07',
            '08', '09', '10', '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', '21', '22', '23',
            '24', '25', '26', '27', '28', '29', '30', '31',
            '32', '33', '34', '35', '36', '37', '38', '39',
            '40', '41', '42', '43', '44', '45', '46', '47',
            '48', '49', '50', '51', '52', '53', '54', '55',
            '56', '57', '58', '59')
    minute.set(minutes[0])

    mins = OptionMenu(frame, minute, *minutes)
    mins.pack(side=LEFT)
    
    set_alarm_button = Button(root,text="Set Alarm",font=("Helvetica 15"),command=start_alarm)
    set_alarm_button.pack(pady=20)
    stop_alarm_button = Button(root,text="Stop Alarm",font=("Helvetica 15"),command=stop_alarm)
    stop_alarm_button.pack(pady=20)
    snooze_button = Button(root,text="Snooze",font=("Helvetica 15"),command=snooze)
    snooze_button.pack(pady=20)
    
    # Execute Tkinter
    root.mainloop()

# driver function
def main():
    init_gui()

if __name__ == '__main__':
    main()