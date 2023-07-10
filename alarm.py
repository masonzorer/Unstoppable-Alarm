# file for python alarm clock
from pygame import mixer
from tkinter import *
from threading import *
import datetime
import time

# Global GUI variables
root = Tk()
set_alarm_button = None
snooze_button = None
hour = StringVar(root)
minute = StringVar(root)

# Global alarm variables
snoozes_left = 3
mixer.init()
alarm_thread = None
alarm_stop_event = Event()
music_playing = False

# alarm function
def alarm():
    alarm_time = f"{hour.get()}:{minute.get()}:00"
    while alarm_stop_event.is_set() == False:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        if now == alarm_time:
            print("Wake up!")
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
    global alarm_thread
    if alarm_thread == None:
        return
    if music_playing:
        stop_sound()

    # stop current alarm
    alarm_stop_event.set()
    alarm_thread.join()
    alarm_thread = None

# play music
def play_sound():
    global music_playing
    if music_playing:
        return
    mixer.music.load('FitnessGram.mp3')
    mixer.music.play()
    music_playing = True

# stop music
def stop_sound():
    global music_playing
    if not music_playing:
        return
    mixer.music.stop()
    music_playing = False

# Functions to handle the snooze button press
def snooze_alarm():
    time.sleep(5)
    if alarm_thread == None:
        return
    play_sound()

def snooze():
    global alarm_thread, snoozes_left
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

# Function to disable the close button
def disable_event():
    pass

# Function to lock the alarm changes
def lock_alarm():
    global set_alarm_button
    if set_alarm_button['state'] == NORMAL:
        set_alarm_button['state'] = DISABLED

# Lock snooze button
def lock_snooze():
    global snooze_button
    if snooze_button['state'] == NORMAL:
        snooze_button['state'] = DISABLED

# GUI initialization function
def init_gui():
    global set_alarm_button, snooze_button
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