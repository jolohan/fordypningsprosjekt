users = {}

def add_all_data_from_year(year=2016):
    global users
    csvfile = open('data/trips_'+str(year)+'.csv', 'r').readlines()
    for i, row in enumerate(csvfile):
        if ()
        if (i == 0):
            continue
        user_ID = row.split(',')[1]
        if (user_ID not in users.keys()):
            users[user_ID] = []
        users[user_ID].append(row)

add_all_data_from_year(year=2016)
add_all_data_from_year(year=2017)

for user in users.keys():
    filename = user.rstrip()
    filename = filename[1:-1]
    open('data/trips_by_user/user_' + str(filename) + '.csv', 'w+').writelines(users[user])
