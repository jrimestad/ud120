#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

mydict = {}
print "People:", len(enron_data)
print "Features:", len(enron_data["SKILLING JEFFREY K"])

pois = [person for person in enron_data if enron_data[person]["poi"] == True]

print "POI's:", len(pois)

poi_names = []
with open("../final_project/poi_names.txt", "r") as poi_names_file:
    for line in poi_names_file:
        if "(" == line[0]:
            poi_names.append(line)  

print "Total pois:", len(poi_names)

print "James Prentice stock options", enron_data["PRENTICE JAMES"]["total_stock_value"]

print "Wesley Colwell emails to pois", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "Jeffrey K Skilling stock value", enron_data["SKILLING JEFFREY K"]["total_stock_value"]

payments = {}
for poi in pois:
    payments[poi] = enron_data[poi]["total_payments"]
max_man = max(payments, key=lambda x: payments[x])
print "Max total_payments: ", max_man, " amount: ", payments[max_man]

print "Valid salary: ", len([x for x in enron_data if enron_data[x]["salary"] != "NaN"])
print "Valid email: ", len([x for x in enron_data if not "NaN" in enron_data[x]["email_address"]])
