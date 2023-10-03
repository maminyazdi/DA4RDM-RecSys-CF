from frictionless import validate

#with open('input/21-06-2022.csv') as file:

report = validate('input/21-06-2022.csv')
print(report.to_summary())