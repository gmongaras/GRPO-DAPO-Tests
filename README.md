Just playing around a little with [GRPO](https://arxiv.org/pdf/2402.03300) and [DAPO](https://arxiv.org/abs/2503.14476)

May also add some RLHF here if I can get it working. Seems the current code works. I just only have it on one GPU making it really slow. But it does make progress it seems.

```
Question: Maya packed a gift box for her cousin. She first placed the empty box on a scale and added enough cookies to make the box weigh 3 pounds. Then, she added enough chocolate bars to double the weight. After that, she added another 3 pounds of cookies. Finally, she added enough candy canes to triple the weight of the box at that point. What was the final weight of the box, in pounds?assistant

To solve this problem, let's break it down step by step.

1. Maya first placed an empty box on the scale and added enough cookies to make it weigh 3 pounds. This means the box now weighs 3 pounds with the cookies.

2. Then, she added enough chocolate bars to double the weight. Since the box weighed 3 pounds with the cookies, doubling it means it now weighs 3 * 2 = 6 pounds.

3. After that, she added another 3 pounds of cookies. Now, the weight is 6 (from the chocolate bars) + 3 (new cookies) = 9 pounds.

4. Finally, she added enough candy canes to triple the weight of the box at that point. Since the box weighed 9 pounds at this point, tripling it means it now weighs 9 * 3 = 27 pounds.

Therefore, the final weight of the box is 27 pounds.

{
    "solution": 27
}
```