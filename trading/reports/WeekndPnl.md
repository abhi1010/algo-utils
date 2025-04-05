# How to generate weekend reports

The following is a script I used for quickly checking how much profit I was making per day and symbol.

However it had minor niggles, like:

1. It can't track manual positions I took to clear up
2. There were days when the code actually missed updating DB, due to bugs. They are forever lost.

ANyways, I exported the output to sheets, and created a pivot table by date - month and week number to see how this strategy was performning.

```sql

DROP TABLE IF EXISTS BuyTrades;
DROP TABLE IF EXISTS SellTrades;

CREATE TEMP TABLE BuyTrades AS
SELECT
    date,
    tag,
    ticker,
    json_extract(additional_attribute, '$.tradedQuantity') AS quantity,
    json_extract(additional_attribute, '$.tradedPrice') AS price
FROM transactions
WHERE json_extract(additional_attribute, '$.transactionType') = 'BUY'
     and tag='weekend'
     and ( txn_type='trades' or txn_type='tradebook' )
ORDER BY date;

CREATE TEMP TABLE SellTrades AS
SELECT
    date,
    tag,
    ticker,
    json_extract(additional_attribute, '$.tradedQuantity') AS quantity,
    json_extract(additional_attribute, '$.tradedPrice') AS price
FROM transactions
WHERE json_extract(additional_attribute, '$.transactionType') = 'SELL'
     and tag='weekend'
     and ( txn_type='trades' or txn_type='tradebook' )
ORDER BY date;



WITH FIFO_Matches AS (
    SELECT
        BuyTrades.date AS buy_date,
        SellTrades.date AS sell_date,
        BuyTrades.tag,
        BuyTrades.ticker,
        MIN(BuyTrades.quantity, SellTrades.quantity) AS matched_quantity,
        BuyTrades.price AS buy_price,
        SellTrades.price AS sell_price,
        (SellTrades.price - BuyTrades.price) * MIN(BuyTrades.quantity, SellTrades.quantity) AS profit_loss,
        ROW_NUMBER() OVER (PARTITION BY SellTrades.date, SellTrades.ticker ORDER BY BuyTrades.date) AS match_order
    FROM
        BuyTrades
        JOIN SellTrades ON BuyTrades.tag = SellTrades.tag AND BuyTrades.ticker = SellTrades.ticker
    WHERE
        BuyTrades.date <= SellTrades.date
)
SELECT
    sell_date,
    tag,
    ticker,
    SUM(matched_quantity) AS total_sold_quantity,
    SUM(profit_loss) AS total_profit_loss
FROM
    FIFO_Matches
GROUP BY
    sell_date, tag, ticker
ORDER BY
    sell_date;
```


## How to get a print out of trades, in a table format


```sql
SELECT
    date,
    json_extract(additional_attribute, '$.tradingSymbol') AS ticker,
    json_extract(additional_attribute, '$.tradedPrice') AS price,
    json_extract(additional_attribute, '$.tradedQuantity') AS quantity,
    json_extract(additional_attribute, '$.transactionType') AS side
FROM transactions
where tag='weekend' and txn_type='trades'
ORDER BY date, ticker;

```

