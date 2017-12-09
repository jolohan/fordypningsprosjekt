# Imports the Google Cloud client library
from google.cloud import bigquery
import datetime
import numpy as np
from sklearn import preprocessing
import random as rnd
import csv

#########################################
# export GOOGLE_APPLICATION_CREDENTIALS=/Users/johan/PycharmProjects/Fordypningsprosjekt/'UIP students-8ffa90ab95f7.json'
#########################################

# The name of the datasets


# Queries
query1 = 'SELECT COUNT(DISTINCT (ID)) as trips FROM `uip-students.oslo_bysykkel_legacy.trip` ' \
            'WHERE _PARTITIONTIME >= "2017-11-10 00:00:00" ' \
            'AND _PARTITIONTIME < "2017-11-25 00:00:00" LIMIT 100'


# Prepares a reference to the new dataset
# dataset_ref = bigquery_client.dataset(dataset_id)
# dataset = bigquery.Dataset(dataset_ref)
# print('Dataset {} created.'.format(dataset.dataset_id))



class DataManager():



    def __init__(self, training_size_proportion=0.8, test_size_proportion=0.1):
        self.training_set_size = training_size_proportion
        self.test_set_size = test_size_proportion
        self.validation_set_size = 1.0-training_size_proportion-test_size_proportion
        # Instantiates a client
        self.bigquery_client = bigquery.Client()
        self.all_trips = None
        self.all_normalized_trips = None
        self.all_normalized_trips_without_end_station = None
        print("all days are to be collected")
        self.all_days = self.query_all_unique_days_in_trips()
        self.split_days_into_training_test_vaildation()

    # Find all unique days in trips so they can be split into training/test
    def query_all_unique_days_in_trips(self, load_from_local=True):
        unique_days = None
        if (load_from_local):
            with open('query_results/all_trip_dates.csv', 'rt') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                days = [row[0] for row in spamreader][1:]
                unique_days = []
                for row in days:
                    row_split = row.split('-')
                    print(row_split)
                    row_split = [(int)(cell) for cell in row_split]
                    row = datetime.date(row_split[0], row_split[1], row_split[2])
                    unique_days.append(row)
        else:
            query_text = 'SELECT CAST(started_at AS DATE) ' \
                         'FROM `uip-students.oslo_bysykkel_legacy.trip` T ' \
                         'GROUP BY CAST(started_at AS DATE) ' \
                         'LIMIT 10000'
            print("collecting query result...")
            query_result = self.query(query_text)
            print("got query result")
            #days_matrix = [self.split_date_time_object(date[0], remove_time=True) for date in query_result]
            unique_days = return_first_column_of_query_result(query_result)
            unique_days = np.array(unique_days)
        unique_days.sort()
        for i, day in enumerate(unique_days):
            print(day)
        return unique_days

    def split_days_into_training_test_vaildation(self):
        self.training_days = []
        self.training_days_year_week_weekday = []
        self.test_days = []
        self.validation_days = []
        counters = [0, 0, 0]
        for day in self.all_days:
            number = rnd.random()
            if (number<self.training_set_size):
                self.training_days.append(day)
                self.training_days.append(convert_to_year_week_weekday(day))
                counters[0] += 1
            elif (number<self.training_set_size+self.test_set_size):
                self.test_days.append(day)
                counters[1] += 1
            else:
                self.validation_days.append(day)
                counters[2] += 1
        #print(counters)

    def set_training_trips(self):
        self.training_trips = []
        counters = [0, 0]
        for trip in self.all_trips:
            year_week_weekday = trip[2:5]
            print(year_week_weekday)
            print(self.training_days_year_week_weekday[0])
            if year_week_weekday in self.training_days_year_week_weekday:
                self.training_trips.append(trip)
                counters[0] += 1
            else:
                counters[1] += 1

        print(counters)

    def query(self, query):
        client = bigquery.Client()
        query_job = client.query(query)
        return query_job.result()

    def query_all_trips_by_user(self, userID=18340):
        query_text = 'SELECT * FROM `uip-students.oslo_bysykkel_legacy.trip`' \
                'WHERE member_id = ' + str(userID)
        return self.query(query_text)

    def format_trip_query_to_data(self, query_job_result):
        trip_data = list(query_job_result)
        formatted_trips = []
        for trip in trip_data:
            member_ID = trip[1]
            start_station = trip[2]
            end_station = trip[3]
            start_time = trip[5]
            end_time = trip[6]
            if end_station == None:
                pass
            else:
                start_time = self.split_date_time_object(start_time)
                end_time = self.split_date_time_object(end_time)
                formatted_trip = [start_station, end_station]
                formatted_trip += start_time
                formatted_trip += end_time
                formatted_trips.append(formatted_trip)

        #print(len(trip_data))
        formatted_trips = np.array(formatted_trips)
        return formatted_trips

    def split_date_time_object(self, date_time, remove_time=False):
        year = (int)(date_time.year)
        month = (int)(date_time.month)
        day = (int)(date_time.day)
        week = (int)(datetime.date(year, month, day).isocalendar()[1])
        day_of_week = (int)(date_time.weekday())
        if (not remove_time):
            minutes_after_midnight = (int)(date_time.hour*60 + date_time.minute)
            return [year, week, day_of_week, minutes_after_midnight]
        #print(year, month, day, week, day_of_week)
        return [year, week, day_of_week]

    def set_formatted_trips_by_user(self, userID=18340, load_from_file=True):
        query_job_trips = self.query_all_trips_by_user(userID=userID)
        trips = self.format_trip_query_to_data(query_job_result=query_job_trips)
        self.all_trips = trips

    def set_normalized_trips_for_user(self):
        self.set_formatted_trips_by_user()
        self.all_normalized_trips = normalize_data(self.all_trips)
        self.all_normalized_trips_without_end_station = make_copy_deep_at_coulmn_x(self.all_normalized_trips)
        for trip in self.all_normalized_trips_without_end_station:
            trip[1] = 0

def normalize_data(data):
    return preprocessing.scale(data)

def make_copy_deep_at_coulmn_x(twod_numpy_array, deep_copy_index=1):
    new_2d_array = np.copy(twod_numpy_array)
    for i, array in enumerate(twod_numpy_array):
        new_2d_array[i][deep_copy_index] = np.copy(array[deep_copy_index])
    return new_2d_array

def return_first_column_of_query_result(query_result):
    return_array = []
    for row in query_result:
        return_array.append(row[0])
    return return_array

def convert_to_year_week_weekday(day):
    year = (int)(day.year)
    month = (int)(day.month)
    day_number = (int)(day.day)
    week = (int)(datetime.date(year, month, day_number).isocalendar()[1])
    day_of_week = (int)(day.weekday())
    return [year, week, day_of_week]


if __name__ == '__main__':
    data_manager = DataManager()
    trips = data_manager.set_normalized_trips_for_user()
