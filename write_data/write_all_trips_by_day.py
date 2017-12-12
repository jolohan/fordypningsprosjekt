

days = {}

def add_all_data_from_year(year=2016):
	global days
	csvfile = open('data/trips_'+str(year)+'.csv', 'r').readlines()
	for i, row in enumerate(csvfile):
		if (i == 0):
			continue
		day = row.split(',')[3][1:11]
		if (day not in days.keys()):
			days[day] = []
		days[day].append(row)

if __name__ == '__main__':
	print("hey")
	add_all_data_from_year()
	add_all_data_from_year(year=2017)

	for day in days.keys():
		filename = day.rstrip()
		print(filename)
		open('data/trips_by_day/day_' + str(filename) + '.csv', 'w+').writelines(days[day])

