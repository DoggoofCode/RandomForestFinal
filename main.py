from dt import RandomForest

# Sample data: [age, income, purchase]
data = [
    [25, 50000, 0],
    [35, 60000, 1],
    [45, 70000, 1],
    [20, 40000, 0],
    [50, 80000, 1],
    [23, 45000, 0],
    [40, 75000, 1],
    [30, 58000, 0],
    [5, 5000, 0],
    [60, 90000, 1],
    [25, 50000, 0],
    [35, 60000, 1],
    [45, 70000, 1],
    [20, 40000, 0],
    [25, 8_000_000, 1],
    [25, 8_000, 0],
    [23, 45000, 0],
    [40, 75000, 1],
    [30, 58000, 0],
    [55, 85000, 1],
    [60, 900, 0],
    [25, 50000, 0],
    [35, 60000, 1],
    [45, 70000, 1],
    [20, 40000, 0],
]

# Remodeling
best_found_accuracy = 0
best_model = None
hash_table = {}
for i in range(1_000):
    model: RandomForest = RandomForest()
    model.fit(data=data)

    score = 0
    for tc_index, test_case in enumerate(data):
        predicted = model.predict(data[tc_index][:-1])
        necessary = test_case[-1]
        # print(f"\nTestcase index: {tc_index}\n"
        #       f"\tTestcase: {test_case[:-1]}"
        #       f"\tPredicted outcome: {predicted}"
        #       f"\n\tWanted outcome: {necessary}, "
        #       f"\nSuccess:{predicted == necessary}")
        if predicted == necessary:
            score += 1

    accuracy = score / len(data) * 100
    print(f"Total Accuracy: {accuracy}%")

    # hashmap
    if hash_table.get(accuracy):
        hash_table[accuracy] += 1
    else:
        hash_table[accuracy] = 1

    if accuracy > best_found_accuracy:
        best_found_accuracy = accuracy
        best_model = model
        print(f"New Best!: {best_found_accuracy}%")

print(f"Best Accuracy: {best_found_accuracy}%")
print(hash_table)
print(best_model.predict([37, 62_000]))
