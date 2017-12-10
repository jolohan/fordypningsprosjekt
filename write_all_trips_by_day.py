
def write_all_trips_by_day_by_year(year=2016) -> object:
	csvfile = open( 'data/trips_' + str(year) +'.csv', 'r').readlines()
	filename = None
	old_date = csvfile[1].split(",")[3][1:11]
	rows_to_save = []
	for i, row in enumerate(csvfile):
		if (i == 0):
			continue
		new_date = row.split(",")[3][1:11]
		if (new_date != old_date):
			#print(new_date)
			# 03.04.2017
			# filename = old_date[6:]
			# filename += old_date[2:6]
			# filename += old_date[:2]
			filename = old_date
			open('data/trips_by_day/trips_' + str(filename) + '.csv', 'w+').writelines(rows_to_save)
			rows_to_save = []
		rows_to_save.append(row)

		old_date = new_date

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

