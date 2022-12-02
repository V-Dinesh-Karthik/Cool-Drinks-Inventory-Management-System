import sqlite3 as sql
import pandas as pd


# connect to database
conn = sql.connect("base.db",check_same_thread=False)

# create table


def create():
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS ITEMS(Time DATE, Name TEXT, Count INT)''')
    conn.commit()

# add data to the table


def Insert(date, text, count):
    conn.execute('''INSERT INTO ITEMS VALUES(?,?,?)''',
                 (date, text, count))
    conn.commit()


# read contents of the data
def read():
    result = []
    state = conn.execute('''SELECT * FROM ITEMS''')
    data = state.fetchall()
    for row in data:
        result.append(row)
    df = pd.DataFrame(result,columns=['time','name','count'])
    return df

def read_():
    l = []
    state = conn.execute('''SELECT Name,SUM(Count) FROM ITEMS GROUP BY Name''')
    data = state.fetchall()
    for row in data:
        l.append(row)
    df = pd.DataFrame(l,columns=['','count'])

    return df


# truncate table


def _delete():
    conn.execute('''DELETE FROM ITEMS''')
    conn.commit()


if __name__ == '__main__':
    create()
