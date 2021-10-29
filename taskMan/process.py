import wmi
import time
import csv

from datetime import datetime

f = wmi.WMI()

def isMalicious(process):
	process = process.lower()
	if 'discord' in process or '.py' in process or 'whatsapp' in process:
		return True
	else:
		return False

def sendProcessInfoAcrossNetwork(process):
	print(process)

current_running_processes = []

with open('log.csv', 'a', newline = '') as file:
	writer = csv.writer(file)
	writer.writerow(['S.No.','Process_Name', 'Time'])

i = 1

for process in f.Win32_Process():
	proc_name = process.Name
	if proc_name not in current_running_processes:
		current_running_processes.append(proc_name)
		if isMalicious(proc_name):
			now = datetime.now()
			current_time = now.strftime("%H:%M:%S")
			with open('log.csv', 'a', newline = '') as file:
				writer = csv.writer(file)
				writer.writerow([str(i), proc_name, current_time])
				i += 1

while True:
	#time.sleep(5)
	print("Checking againn!")
	for process in f.Win32_Process():
		proc_name = process.Name
		if proc_name not in current_running_processes:
			now = datetime.now()
			current_time = now.strftime("%H:%M:%S")
			current_running_processes.append(proc_name)
			with open('log.csv', 'a', newline = '') as file:
				writer = csv.writer(file)
				writer.writerow([str(i), proc_name, current_time])
				i += 1
