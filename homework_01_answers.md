# ML Zoomcamp 2026 — Homework 1 Answers

Source: [Homework 1](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/01-intro/homework.md)

The calculations are reproduced in `homework_01_solution.ipynb` using the pinned dataset `car_fuel_efficiency_2026.csv`.

## Answers

1. **Pandas version:** `3.0.1`
2. **Records count:** **10,000**
3. **Fuel types:** **3** (`Diesel`, `Gasoline`, and `Hybrid`)
4. **Columns with missing values:** **2** (`horsepower` and `acceleration`)
5. **Maximum fuel efficiency for cars from Asia:** **41.2 MPG**
6. **Did the horsepower median change after filling missing values with the mode?** **Yes, it decreased.**
   - Median before filling: `254.0`
   - Mode used for filling: `252.0`
   - Median after filling: `252.0`
7. **Sum of the elements of `w`:** `0.36919696904925486`, so select **0.369**.

## Q7 intermediate values

```text
X =
[[4240, 1996],
 [4310, 1992],
 [4230, 1997],
 [4420, 1978],
 [3950, 2007],
 [4750, 2021],
 [4450, 2019]]

XTX =
[[131950500, 60750580],
 [ 60750580, 28041424]]

w = [0.13644777, 0.23274920]
sum(w) = 0.36919696904925486
```
