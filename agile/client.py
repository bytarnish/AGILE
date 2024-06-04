import re
import sqlite3


class SQLClient(object):
    def __init__(self, db_path='product_qa.db'):
        # Define the database connection here
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def _add_not_null(self, sql, group):
        # add not null constraint to "order by"
        pattern = re.compile(r'order by (.*?) ', flags=re.IGNORECASE)
        matches = pattern.findall(sql)
        if len(matches) > 0:
            sql = re.sub(re.compile(r'order by', flags=re.IGNORECASE), "AND {} IS NOT NULL ORDER BY".format(matches[0]), sql)
            sql = sql.replace("{} AND".format(group), "{} WHERE".format(group))
        return sql

    def _select_asin(self, sql, group):
        # force to select asin
        sql = sql.replace("product_id", "asin")
        sql = sql.strip().strip(".")
        return re.sub(re.compile(r'^select (.*?) from \`' + group + '`', flags=re.IGNORECASE), "SELECT asin, title FROM `{}`".format(group), sql)

    def sql_query(self, sql, group, return_title=False):
        sql = self._select_asin(sql, group)
        sql_not_null = self._add_not_null(sql, group)
        try:
            # first execute sql with "not null"
            self.cursor.execute(sql_not_null)
            result = self.cursor.fetchall()
            if return_title:
                result = [list(i) for i in result]
            else:
                result = [i[0] for i in result]
            return result
        except Exception as e:
            print(e)
            pass
        try:
            # execute original sql
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            if return_title:
                result = [list(i) for i in result]
            else:
                result = [i[0] for i in result]
            return result
        except:
            return "err"


if __name__ == "__main__":
    sql_client = SQLClient()
    group = "headphones"
    sql = "SELECT asin FROM headphones WHERE price > (SELECT AVG(price) FROM headphones) AND price IS NOT NULL ORDER BY price ASC limit 4;"
    asin_list = sql_client.sql_query(sql, group)
    print(asin_list)
