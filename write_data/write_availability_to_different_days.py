days = {}
path = 'data/station_availability'

def add_all_data_from_year():
	global days
	csvfile = open(path+'.csv', 'r').readlines()[1:]
	for i, row in enumerate(csvfile):
		day = row.split(',')[6][:10]
		day = '.'.join(day.split('-'))
		if (day not in days.keys()):
			print(len(days.keys()))
			days[day] = []
		days[day].append(row)

if __name__ == '__main__':
	print("hey")
	add_all_data_from_year()
	for day in days.keys():
		filename = day
		print(day)
		open(path + '/day_' + str(filename) + '.csv', 'w+').writelines(days[day])