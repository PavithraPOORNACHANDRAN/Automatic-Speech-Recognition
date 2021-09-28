# Getting first hypothesis
# AalaElKhani_2016X.csv , AaronHuey_2010X.csv
import csv
files=["data.csv","AaronHuey_2010X.csv"]
content_file=list()
# iterate files in list
for  file in files:
    print(file)
    print("-------------------------------")
    content = list()
    begin=list()
    sorted_content=list()
    with open(file, 'r') as inp:
        # iterate each row in each input file
        for row in csv.reader(inp):

            if row[2]!="begin":
                begin.append(row[2])
            if row[2] == "begin" or row[4] == "1":
                content.append(row)
    # here sort begin list
    begin.sort()
    # iterate begin values
    for value in begin:
        # iterate content values
        for data in content:
            # for each value in data if exist then copy to sorted_content
            if value in data:
                sorted_content.append(data)
                break
    # appending both file sorted content to 1 single list content_file
    for data in content:
        content_file.append(data)
    print(len(begin))
    print(len(content))
    print(len(sorted_content))

print(content_file)
with open('useinput.csv','w') as file:
    writer = csv.writer(file)
    writer.writerow(["sent_id", "sentence"])
    for row in content_file:
        if row[1]  == "sent_id":
            continue
        writer.writerow([row[1],row[5]])

with open('train_ids.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["Current_Id", "Prevsent_id"])
        i=0
        # iterate content_flie
        for i in range(len(content_file)):
            if content_file[i][1] == "sent_id":
                continue
            # for len //2 we are checking for 2nd content data whether sent_id is not copying 2 times
            if i==0 or len(content_file)//2==i or content_file[i-1][1] == "sent_id":
                # in 1 st row, current_id and prev_id are same
                writer.writerow([content_file[i][1], content_file[i][1]])
            else:
                # Current_id , Current_id - 1
                writer.writerow([content_file[i][1], content_file[i-1][1]])