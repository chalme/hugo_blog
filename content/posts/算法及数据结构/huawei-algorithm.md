---
title: "Huawei Algorithm"
date: 2024-07-15T16:10:44+08:00
draft: false
tags:
  - 算法与数据结构
ShowToc: true
---
## 滑动窗口

### 盛最多水的容器 <https://leetcode.cn/problems/container-with-most-water/>

```java
class Solution {
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int res = 0;
        while (l < r) {
            int area = Math.min(height[l], height[r]) * (r - l);
            res = Math.max(res, area);
            if (height[l] <= height[r]) {
                l++;
            } else {
                r--;
            }
        }
        return res;
    }
}
```

### 接雨水

- dp 每一个i的最多接的雨水， max（leftMax(i), rightMax(i)) -height(i)
- 双指针
- ![](https://img.chalme.top/c/20240710-fc4cd084e3d39fb3-202407101554324.png)

```java
class Solution {
    public int trap(int[] height) {
        int len = height.length;
        int[] left = new int[len];
        int[] right = new int[len];
        left[0] = height[0];
        for (int i = 1; i < len; i++) {
            left[i] = Math.max(left[i - 1], height[i]);
        }
        right[len - 1] = height[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            right[i] = Math.max(right[i + 1], height[i]);
        }
        int res = 0;
        for (int i = 0; i < len; i++) {
            res += Math.min(left[i], right[i]) - height[i];
        }
        return res;
    }
}

class Solution {
    public int trap(int[] height) {
        int len = height.length;
        int left = 0, right = len -1;
        int leftMax = 0, rightMax = 0;
        int res = 0;
        while (left <= right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            if (height[left] <= height[right]) {
                res += leftMax - height[left];
                left ++;
            } else {
                res += rightMax - height[right];
                right --;
            }
        }
        return res;
    }
}
```

### 无重复字符的最长子串

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int len = s.length();
        if (len <= 1) {
            return len;
        }
        HashSet<Character> set = new HashSet<>();
        int l = 0, r = 1;
        int res = 0;
        set.add(s.charAt(l));
        while (r < len) {
            while (r < len && !set.contains(s.charAt(r))) {
                set.add(s.charAt(r));
                r++;
            }
            res = Math.max(res, r - l);
            set.remove(s.charAt(l));
            l++;
        }
        return res;
    }
}
```

## DFS/BFS

### 200 岛屿数量

```java
class Solution {
    public int numIslands(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '2';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
}
```

### 不同的二叉搜索树

```java
class Solution {

    Map<Integer, Integer> map = new HashMap<>();

    public int numTrees(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }
        if (map.containsKey(n)) {
            return map.get(n);
        }

        int count = 0;
        for (int i = 1; i <= n; i++) {
            int leftNum = numTrees(i - 1);
            int rightNum = numTrees(n - i);
            count += leftNum * rightNum;
        }
        map.put(n, count);

        return count;

    }
}
```

## 动态规划

### 122. 买卖股票的最佳时机 II

### 123. 买卖股票的最佳时机 III

### 213 打家劫舍II

```java
// 121. 买卖股票的最佳时机
// dp
class Solution {
    public int maxProfit(int[] prices) {
        int cost = Integer.MAX_VALUE, profit = 0;
        for (int price : prices) {
            cost = Math.min(cost, price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }
}


// 122. 买卖股票的最佳时机 II
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i=0; i<n; i++) {
            if (i==0) {
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
            } else {
                dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1]+prices[i]);
                dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0]-prices[i]);
            }
        }
        return dp[n-1][0];

    }
}
// 123
// https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        // 边界判断, 题目中 length >= 1, 所以可省去
        if (prices.length == 0) return 0;

        /*
         * 定义 5 种状态:
         * 0: 没有操作, 1: 第一次买入, 2: 第一次卖出, 3: 第二次买入, 4: 第二次卖出
         */
        int[][] dp = new int[len][5];
        dp[0][1] = -prices[0];
        // 初始化第二次买入的状态是确保 最后结果是最多两次买卖的最大利润
        dp[0][3] = -prices[0];

        for (int i = 1; i < len; i++) {
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i][3] + prices[i]);
        }

        return dp[len - 1][4];
    }
}

// 213
class Solution {
    public int rob(int[] nums) {
        int len = nums.length;
        if (len == 1) {
            return nums[0];
        }

        // dp1 取0， 不取len-1， dp2 取n-1， 不取0
        int[] dp1 = new int[len];
        int[] dp2 = new int[len];
        dp1[0] = nums[0];
        dp1[1] = Math.max(nums[0], nums[1]);
        dp2[1] = nums[1];
        for (int i = 2; i < len; i++) {
            dp1[i] = Math.max(dp1[i - 1], dp1[i - 2] + nums[i]);
            dp2[i] = Math.max(dp2[i - 1], dp2[i - 2] + nums[i]);
        }
        return Math.max(dp1[len - 2], dp2[len - 1]);
    }
}
```

## 递归：LeetCode70、112、509

```java
//70 爬楼底
class Solution {
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int[] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;
        for (int i=2; i<n; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n-1];

    }
}

// 112. 路径总和
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }
}

// 509. 斐波那契数  打表、递归、循序
class Solution {
    public int fib(int n) {
        if (n < 2) {
            return n;
        }
        int p = 0, q = 0, r = 1;
        for (int i = 2; i <= n; ++i) {
            p = q;
            q = r;
            r = p + q;
        }
        return r;
    }
}
```

## 分治：LeetCode23、169、240

```java
// 23. 合并 K 个升序链表
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int len = lists.length;
        if (len == 0) {
            return null;
        }

        PriorityQueue<ListNode> queue = new PriorityQueue<>(len, (l1, l2) -> l1.val - l2.val);

        for (int i = 0; i < len; i++) {
            if (lists[i] != null) {
                queue.add(lists[i]);
            }
        }
        ListNode dummy = new ListNode();
        ListNode tmp = dummy;
        while (!queue.isEmpty()) {
            ListNode node = queue.poll();
            if (node.next != null) {
                queue.add(node.next);
            }
            tmp.next = node;
            tmp = node;
        }
        return dummy.next;

    }
}
//169. 多数元素

class Solution {
    /**
     * 1. hash O(n)/O(n)
     * 2. 排序  O(nlgn)/O(nlogn)
     * 3. 投票  O(n)/O(n)
     * 分治
     */
    public int majorityElement(int[] nums) {
        Integer candidate = null;
        int count = 0;
        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            count += (candidate == num) ? 1 : -1;
        }
        return candidate;
    }
}

// 240. 搜索二维矩阵 II
class Solution {
    /**
     * z 型搜索 O(n+m)
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int n = matrix.length, m = matrix[0].length;
        int x = 0, y = m - 1;
        while (x < n && y >= 0) {
            if (matrix[x][y] == target) {
                return true;
            } else if (matrix[x][y] < target) {
                x++;
            } else {
                y--;
            }
        }
        return false;
    }
}
```

## 单调栈：LeetCode84、85、739、503

```java
// 84. 柱状图中最大的矩形
class Solution {
    public int largestRectangleArea(int[] heights) {
        int[] newHeights = Arrays.copyOf(heights, heights.length + 1);
        int maxArea = 0;
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < newHeights.length; i++) {
            while (!stack.isEmpty() && newHeights[stack.peek()] > newHeights[i]) {
                int height = newHeights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            stack.push(i);
        }
        return maxArea;
    }
}

// 85. 最大矩形
class Solution {
    /**
     * left[i][j] 代表 第 i 行第 j 列元素的左边连续 1 的数量， 暴力
     */
    public int maximalRectangle(char[][] matrix) {
        int n = matrix.length;
        if (n == 0) {
            return 0;
        }
        int m = matrix[0].length;

        int[][] left = new int[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = j == 0 ? 1 : left[i][j - 1] + 1;
                }
            }
        }
        int res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] == '0')
                    continue;
                int width = left[i][j];
                res = Math.max(width, res);
                for (int k = i - 1; k >= 0; k--) {
                    width = Math.min(left[k][j], width);
                    res = Math.max(res, width * (i - k + 1));
                }
            }
        }
        return res;
    }
}

// 739. 每日温度
class Solution {
    /**
     * 单调递减栈
     */
    public int[] dailyTemperatures(int[] temperatures) {
        int len = temperatures.length;
        int[] ans = new int[len];
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < len; i++) {
            while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
                int prevIndex = stack.pop();
                ans[prevIndex] = i - prevIndex;
            }
            stack.push(i);
        }
        return ans;

    }
}

// 530. 二叉搜索树的最小绝对差
class Solution {
    int pre;
    int ans;

    public int getMinimumDifference(TreeNode root) {
        pre = -1;
        ans = Integer.MAX_VALUE;
        dfs(root);
        return ans;

    }

    void dfs(TreeNode root) {
        if (root == null) {
            return;
        }

        dfs(root.left);
        if (pre == -1) {
            pre = root.val;
        } else {
            ans = Math.min(ans, root.val - pre);
            pre = root.val;
        }
        dfs(root.right);
    }
}
```

## 并查集：LeetCode547、200、684

```java
// 547. 省份数量
// dfs
class Solution {
    public int findCircleNum(int[][] isConnected) {
        int cities = isConnected.length;
        boolean[] visited = new boolean[cities];
        int res = 0;
        for (int i=0; i<cities; i++) {
            if (!visited[i]) {
                dfs(isConnected, visited, cities, i);
                res ++;
            }
        }
        return res;
    }

    void dfs(int[][] isConnected, boolean[] visited, int cities, int i) {
        for (int j=0; j<cities; j++) {
            if (isConnected[i][j] == 1 && !visited[j]) {
                visited[j] = true;
                dfs(isConnected, visited, cities, j);
            }
        }
    }
}
// 并查集
class Solution {
    public int findCircleNum(int[][] isConnected) {
        int cities = isConnected.length;
        int[] parent = new int[cities];

        for (int i = 0; i < cities; i++) {
            parent[i] = i;
        }

        for (int i = 0; i < cities; i++) {
            for (int j = i + 1; j < cities; j++) {
                if (isConnected[i][j] == 1) {
                    union(parent, i, j);
                }
            }
        }
        int provinces = 0;
        for (int i = 0; i < cities; i++) {
            if (parent[i] == i) {
                provinces++;
            }
        }
        return provinces;
    }

    public void union(int[] parent, int index1, int index2) {
        parent[find(parent, index1)] = find(parent, index2);
    }

    public int find(int[] parent, int index) {
        if (parent[index] != index) {
            parent[index] = find(parent, parent[index]);
        }
        return parent[index];
    }
}

// 200. 岛屿数量 dfs
class Solution {
    public int numIslands(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '2';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
}

// 684. 冗余连接
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        int[] parent = new int[n + 1];
        for (int i=1; i<n+1; i++) {
            parent[i] = i;
        }

        for (int i=0; i<n; i++) {
            int[] edge = edges[i];
            int node1 = edge[0], node2 = edge[1];
            if (find(parent, node1) != find(parent, node2)) {
                union(parent, node1, node2);
            } else {
                return edge;
            }
        }
        return new int[0];
    }

    public void union(int[] parent, int index1, int index2) {
        parent[find(parent, index1)] = find(parent, index2);
    }

    public int find(int[] parent, int index) {
        if (parent[index] != index) {
            parent[index] = find(parent, parent[index]);
        }
        return parent[index];
    }
}
```

## 滑动窗口：LeetCode209、3、1004、1208

```java
// 209. 长度最小的子数组
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length;
        int left = 0, right = 0;
        int res = Integer.MAX_VALUE;
        int sum = 0;
        while (right < n) {
            sum += nums[right];
            while (sum >= target) {
                if (right - left + 1 < res) {
                    res = right - left + 1;
                }
                sum -= nums[left];
                left++;
            }
            right++;
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }
}

// 3. 无重复字符的最长子串
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int len = s.length();
        if (len <= 1) {
            return len;
        }
        HashSet<Character> set = new HashSet<>();
        int l = 0, r = 1;
        int res = 0;
        set.add(s.charAt(l));
        while (r < len) {
            while (r < len && !set.contains(s.charAt(r))) {
                set.add(s.charAt(r));
                r++;
            }
            res = Math.max(res, r - l);
            set.remove(s.charAt(l));
            l++;
        }
        return res;
    }
}
// 1004. 最大连续1的个数 III
class Solution {
    public int longestOnes(int[] nums, int k) {
        int len = nums.length;
        int l = 0, r = 0;
        int res = 0;
        int zeroCount = 0;
        while (r < len) {
            if (nums[r] == 0) {
                zeroCount++;
            }
            while (zeroCount > k) {
                if (nums[l] == 0) {
                    zeroCount--;
                }
                l++;
            }
            res = Math.max(res, r - l + 1);
            r++;
        }
        return res;
    }
}

// 1208. 尽可能使字符串相等
class Solution {
    public int equalSubstring(String s, String t, int maxCost) {
        int n = s.length();
        int[] diff = new int[n];
        for (int i = 0; i < n; i++) {
            diff[i] = Math.abs(s.charAt(i) - t.charAt(i));
        }
        int maxLength = 0;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += diff[end];
            while (sum > maxCost) {
                sum -= diff[start];
                start++;
            }
            maxLength = Math.max(maxLength, end - start + 1);
            end++;
        }
        return maxLength;
    }
}
```

## 前缀和：LeetCode724、560、437、1248

```java
// 724. 寻找数组的中心下标
class Solution {
    public int pivotIndex(int[] nums) {
        int sumLeft = 0, sumRight = Arrays.stream(nums).sum();
        for (int i=0; i<nums.length; i++) {
            sumRight -= nums[i];
            if (sumLeft == sumRight) {
                return i;
            }
            sumLeft += nums[i];
        }
        return -1;
    }
}

// 560. 和为 K 的子数组
// 前缀和+暴力
class Solution {
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;

        // 前缀和属猪
        int[] pre = new int[n + 1];
        pre[0] = 0;
        for (int i = 0; i < n; i++) {
            pre[i + 1] = pre[i] + nums[i];
        }

        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (pre[j + 1] - pre[i] == k) {
                    count++;
                }
            }
        }
        return count;
    }
}
// 前缀和 + hash
class Solution {
    public int subarraySum(int[] nums, int k) {
        // key：前缀和，value：key 对应的前缀和的个数
        Map<Integer, Integer> preSumFreq = new HashMap<>();
        preSumFreq.put(0, 1);

        int preSum = 0;
        int count = 0;

        // preSum - (preSum - k) == k
        for (int num : nums) {
            preSum += num;

            if (preSumFreq.containsKey(preSum - k)) {
                count += preSumFreq.get(preSum - k);
            }

            // 然后维护 preSumFreq 的定义
            preSumFreq.put(preSum, preSumFreq.getOrDefault(preSum, 0) + 1);
        }
        return count;

    }
}

// 437. 路径总和 III
// 二叉树前缀和
class Solution {

    Map<Long, Integer> prefixMap;

    public int pathSum(TreeNode root, int targetSum) {
        prefixMap = new HashMap<>();
        prefixMap.put(0L, 1);
        return dfs(root, 0, targetSum);
    }

    public int dfs(TreeNode node, long sum, int target) {
        if (node == null) {
            return 0;
        }

        int res = 0;
        sum += node.val;

        res += prefixMap.getOrDefault(sum - target, 0);
        prefixMap.put(sum, prefixMap.getOrDefault(sum, 0) + 1);

        int left = dfs(node.left, sum, target);
        int right = dfs(node.right, sum, target);

        res = res + left + right;

        prefixMap.put(sum, prefixMap.get(sum) - 1);
        return res;
    }
}

// 1248. 统计「优美子数组」
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {

        Map<Integer, Integer> preMap = new HashMap<>();
        preMap.put(0, 1);
        // sum - (sum - k) = k
        int sum = 0;
        int cnt = 0;
        for (int num : nums) {
            sum += num&1;

            cnt += preMap.getOrDefault(sum - k, 0);

            preMap.put(sum, preMap.getOrDefault(sum, 0) + 1);

        }
        return cnt;
    }
}
```

## 差分：LeetCode1094

```java
// 1094. 拼车
class Solution {
    public boolean carPooling(int[][] trips, int capacity) {
        int[] nums = new int[1010];
        for (int[] t: trips) {
            int c = t[0], a = t[1], b = t[2];
            nums[a+1] += c; nums[b+1] -= c;
        }
        for (int i=1; i<= 1000; i++) {
            nums[i] += nums[i-1];
            if (nums[i] > capacity) return false;
        }
        return true;
    }
}
```

## 拓扑排序：LeetCode210

```java
// 210. 课程表 II
// BFS https://leetcode.cn/problems/course-schedule-ii/solutions/221559/java-jian-dan-hao-li-jie-de-tuo-bu-pai-xu-by-kelly/
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses == 0) return new int[0];
        int[] inDegrees = new int[numCourses];
        // 建立入度表
        for (int[] p : prerequisites) { // 对于有先修课的课程，计算有几门先修课
            inDegrees[p[0]]++;
        }
        // 入度为0的节点队列
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < inDegrees.length; i++) {
            if (inDegrees[i] == 0) queue.offer(i);
        }
        int count = 0;  // 记录可以学完的课程数量
        int[] res = new int[numCourses];  // 可以学完的课程
        // 根据提供的先修课列表，删除入度为 0 的节点
        while (!queue.isEmpty()){
            int curr = queue.poll();
            res[count++] = curr;   // 将可以学完的课程加入结果当中
            for (int[] p : prerequisites) {
                if (p[1] == curr){
                    inDegrees[p[0]]--;
                    if (inDegrees[p[0]] == 0) queue.offer(p[0]);
                }
            }
        }
        if (count == numCourses) return res;
        return new int[0];
    }
```

## 字符串：LeetCode5、20、43、93

```java
// 5. 最长回文子串
// 中心扩展
class Solution {
    public String longestPalindrome(String s) {
        if (s.length() <= 1) {
            return s;
        }

        int left = 0, right = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (right - left < len) {
                left = i - (len - 1) / 2;
                right = i + len / 2;
            }
        }
        return s.substring(left, right + 1);
    }

    public int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }
}
// dp
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 1) {
            return s;
        }
        int len = s.length();
        int maxLeft = 0;
        int maxRight = 0;
        int maxLen = 0;
        boolean dp[][] = new boolean[len][len];
        for (int r = 1; r<len; r++) {
            for (int l=0; l<r; l++) {
                if (s.charAt(l) == s.charAt(r) && (r-l <=2 || dp[l+1][r-1])) {
                    dp[l][r] = true;
                    if (r - l + 1 > maxLen) {
                        maxLen = r - l + 1;
                        maxLeft = l;
                        maxRight = r;
                    }
                }
            }
        }
        return s.substring(maxLeft, maxRight + 1);
    }
}

// 20  栈处理
class Solution {
    public boolean isValid(String s) {
        Map<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put('}', '{');
        map.put(']', '[');

        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!stack.isEmpty() && stack.peek() == map.get(c)) {
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }
}

// 43. 字符串相乘
class Solution {
    /**
    * 计算形式
    *    num1
    *  x num2
    *  ------
    *  result
    */
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        // 保存计算结果
        String res = "0";

        // num2 逐位与 num1 相乘
        for (int i = num2.length() - 1; i >= 0; i--) {
            int carry = 0;
            // 保存 num2 第i位数字与 num1 相乘的结果
            StringBuilder temp = new StringBuilder();
            // 补 0
            for (int j = 0; j < num2.length() - 1 - i; j++) {
                temp.append(0);
            }
            int n2 = num2.charAt(i) - '0';

            // num2 的第 i 位数字 n2 与 num1 相乘
            for (int j = num1.length() - 1; j >= 0 || carry != 0; j--) {
                int n1 = j < 0 ? 0 : num1.charAt(j) - '0';
                int product = (n1 * n2 + carry) % 10;
                temp.append(product);
                carry = (n1 * n2 + carry) / 10;
            }
            // 将当前结果与新计算的结果求和作为新的结果
            res = addStrings(res, temp.reverse().toString());
        }
        return res;
    }

    /**
     * 对两个字符串数字进行相加，返回字符串形式的和
     */
    public String addStrings(String num1, String num2) {
        StringBuilder builder = new StringBuilder();
        int carry = 0;
        for (int i = num1.length() - 1, j = num2.length() - 1;
             i >= 0 || j >= 0 || carry != 0;
             i--, j--) {
            int x = i < 0 ? 0 : num1.charAt(i) - '0';
            int y = j < 0 ? 0 : num2.charAt(j) - '0';
            int sum = (x + y + carry) % 10;
            builder.append(sum);
            carry = (x + y + carry) / 10;
        }
        return builder.reverse().toString();
    }
}

// 93. 复原 IP 地址
class Solution {
    public List<String> restoreIpAddresses(String s) {
        int len = s.length();
        List<String> res = new ArrayList<>();
        if (len < 4 || len > 12) {
            return res;
        }
        Deque<String> tmp = new LinkedList<>();
        dfs(s, len, 0, 4, tmp, res);
        return res;
    }

    private void dfs(String s, int len, int start, int left, Deque<String> path, List<String> res) {
        if (start == len) {
            if (left == 0) {
                res.add(String.join(".", path));
            }
            return;
        }

        for (int i=start; i<start + 3; i++) {
            if (i >= len) {
                break;
            }
            if (left * 3 < len - i) {
                continue;
            }
            String sub = s.substring(start, i+1);
            if (judge(sub)) {
                path.addLast(sub);
                dfs(s, len, i + 1, left - 1, path, res);
                path.removeLast();
            }
        }
    }

    private boolean judge(String s) {
        int len = s.length();
        if (len > 1 && s.charAt(0) == '0') {
            return false;
        }
        int t = Integer.valueOf(s);
        return t >= 0 && t <= 255;
    }
}
```

## 二分查找：LeetCode33、34

```java
// 33. 搜索旋转排序数组
class Solution {
    public int search(int[] nums, int target) {
        int len = nums.length;
        if (len == 0) {
            return -1;
        }
        int i = 0, j = len -1;
        while (i <= j) {
            int mid = (i + j) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
                    j = mid - 1;
                } else {
                    i = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[len - 1]) {
                    i = mid + 1;
                } else {
                    j = mid - 1;
                }
            }
        }
        return -1;

    }


}
// 34. 在排序数组中查找元素的第一个和最后一个位置
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int leftIdx = binarySearch(nums, target, true);
        int rightIdx = binarySearch(nums, target, false) - 1;
        if (leftIdx <= rightIdx && rightIdx < nums.length && nums[leftIdx] == target && nums[rightIdx] == target) {
            return new int[]{leftIdx, rightIdx};
        }
        return new int[]{-1, -1};
    }

    public int binarySearch(int[] nums, int target, boolean lower) {
        int left = 0, right = nums.length - 1, ans = nums.length;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target || (lower && nums[mid] >= target)) {
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}
```

## BFS：LeetCode127、139、130、529、815

```java
// 127. 单词接龙

// 139. 单词拆分
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> dict = new HashSet<>(wordDict);
        // dp[i]表示前i个字母是否能被分割
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && dict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}

// 130. 被围绕的区域
// 重点是思路： 从边缘dfs处理
class Solution {
    int n, m;

    public void solve(char[][] board) {
        n = board.length;
        if (n == 0) {
            return;
        }
        m = board[0].length;
        for (int i = 0; i < n; i++) {
            dfs(board, i, 0);
            dfs(board, i, m - 1);
        }
        for (int i = 1; i < m - 1; i++) {
            dfs(board, 0, i);
            dfs(board, n - 1, i);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    public void dfs(char[][] board, int x, int y) {
        if (x < 0 || x >= n || y < 0 || y >= m || board[x][y] != 'O') {
            return;
        }
        board[x][y] = 'A';
        dfs(board, x + 1, y);
        dfs(board, x - 1, y);
        dfs(board, x, y + 1);
        dfs(board, x, y - 1);
    }
}

// 529. 扫雷游戏
// dfs 模拟

// 815. 公交路线（困难）
//优化建图 + 广度优先搜索
```

## DFS&回溯：：LeetCode934、1102、531、533、113、332、337

```java
// 934. 最短的桥
// https://leetcode.cn/problems/shortest-bridge/solutions/1922327/-by-muse-77-j7w5/
class Solution {

    Queue<int[]> queue;
    int[][] direct = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };

    public int shortestBridge(int[][] grid) {

        this.queue = new LinkedList<>();

        boolean find = false;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    find = true;
                    dfs(grid, i, j);
                    break;
                }
            }
            if (find) {
                break;
            }
        }

        int result = 0;
        while (!this.queue.isEmpty()) {
            result++;

            int size = this.queue.size();
            for (int i = 0; i < size; i++) {
                int[] edge = this.queue.poll();
                for (int[] d : direct) {
                    int nex = edge[0] + d[0], ney = edge[1] + d[1];
                    if (judge(nex, ney, grid) && grid[nex][ney] == 0) {
                        this.queue.offer(new int[] { nex, ney });
                        grid[nex][ney] = 2;
                    } else if (judge(nex, ney, grid) && grid[nex][ney] == 1) {
                        return result;
                    }
                }
            }
        }
        return result;
    }

    public void dfs(int[][] grid, int x, int y) {
        if (!judge(x, y, grid) || grid[x][y] == 2)
            return;
        if (grid[x][y] == 0) {
            grid[x][y] = 2;
            this.queue.offer(new int[] { x, y });
            return;
        }
        grid[x][y] = 2;
        dfs(grid, x + 1, y);
        dfs(grid, x - 1, y);
        dfs(grid, x, y + 1);
        dfs(grid, x, y - 1);
    }

    public boolean judge(int x, int y, int[][] grid) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length;
    }
}
// 113. 路径总和 II
class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {

        List<List<Integer>> res = new ArrayList<>();
        dfs(root, targetSum, new ArrayList<Integer>(), 0, res);
        return res;
    }

    public void dfs(TreeNode root, int targetSum, List<Integer> temp, int index, List<List<Integer>> res) {
        if (root == null) {
            return ;
        }
        temp.add(root.val);
        targetSum = targetSum - root.val;
        if (targetSum == 0 && root.left == null && root.right == null) {
            res.add(new ArrayList<>(temp));
            temp.remove(index);
            return;
        }

        dfs(root.left, targetSum, temp, index + 1, res);
        dfs(root.right, targetSum, temp, index + 1,  res);
        temp.remove(index);
    }
}

// 332. 重新安排行程（hard）

// 337. 打家劫舍 III
class Solution {
    public int rob(TreeNode root) {
        int[] res = dfs(root);
        return Math.max(res[0], res[1]); // 根节点选或不选的最大值
    }

    private int[] dfs(TreeNode node) {
        if (node == null) { // 递归边界
            return new int[]{0, 0}; // 没有节点，怎么选都是 0
        }
        int[] left = dfs(node.left); // 递归左子树
        int[] right = dfs(node.right); // 递归右子树
        int rob = left[1] + right[1] + node.val; // 选
        int notRob = Math.max(left[0], left[1]) + Math.max(right[0], right[1]); // 不选
        return new int[]{rob, notRob};
    }
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/house-robber-iii/solutions/2282018/shi-pin-ru-he-si-kao-shu-xing-dppythonja-a7t1/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

## 动态规划：LeetCode213、123、62、63、361、1230

```java
// 213. 打家劫舍 II
class Solution {
    public int rob(int[] nums) {
        int len = nums.length;
        if (len == 1) {
            return nums[0];
        }

        // dp1 取0， 不取len-1， dp2 取n-1， 不取0
        int[] dp1 = new int[len];
        int[] dp2 = new int[len];
        dp1[0] = nums[0];
        dp1[1] = Math.max(nums[0], nums[1]);
        dp2[1] = nums[1];
        for (int i = 2; i < len; i++) {
            dp1[i] = Math.max(dp1[i - 1], dp1[i - 2] + nums[i]);
            dp2[i] = Math.max(dp2[i - 1], dp2[i - 2] + nums[i]);
        }
        return Math.max(dp1[len - 2], dp2[len - 1]);
    }
}

// 123. 买卖股票的最佳时机 III
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        // 边界判断, 题目中 length >= 1, 所以可省去
        if (prices.length == 0) return 0;

        /*
         * 定义 5 种状态:
         * 0: 没有操作, 1: 第一次买入, 2: 第一次卖出, 3: 第二次买入, 4: 第二次卖出
         */
        int[][] dp = new int[len][5];
        dp[0][1] = -prices[0];
        // 初始化第二次买入的状态是确保 最后结果是最多两次买卖的最大利润
        dp[0][3] = -prices[0];

        for (int i = 1; i < len; i++) {
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i][3] + prices[i]);
        }

        return dp[len - 1][4];
    }
}

// 62. 不同路径
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] f = new int[m][n];
        for (int i = 0; i < m; ++i) {
            f[i][0] = 1;
        }
        for (int j = 0; j < n; ++j) {
            f[0][j] = 1;
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                f[i][j] = f[i - 1][j] + f[i][j - 1];
            }
        }
        return f[m - 1][n - 1];
    }
}

// 63. 不同路径 II
class Solution {

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        // 获取网格的行数和列数
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;

        // 如果起点或终点是障碍物，则路径数量为0，直接返回
        if (obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1) {
            return 0;
        }


       int[][] f = new int[m][n];
       f[0][0] = 1;
        for (int i = 1; i < m; ++i) {
            if (obstacleGrid[i][0] == 0 && f[i-1][0] == 1 ) {
                f[i][0] = 1;
            } else {
                f[i][0] = 0;
            }
        }
        for (int j = 1; j < n; ++j) {
            f[0][j] = 1;
            if (obstacleGrid[0][j] == 0 && f[0][j-1] == 1) {
                f[0][j] = 1;
            } else {
                f[0][j] = 0;
            }
        }


        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 0) {
                    f[i][j] = f[i - 1][j] + f[i][j - 1];
                } else {
                    f[i][j] = 0;
                }
            }
        }

        // 返回终点的路径数量
        return f[m - 1][n - 1];
    }
}
```

## 贪心算法：LeetCode55、435、621、452

```java
// 55. 跳跃游戏
public class Solution {
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int rightmost = 0;
        for (int i = 0; i < n; ++i) {
            if (i <= rightmost) {
                rightmost = Math.max(rightmost, i + nums[i]);
                if (rightmost >= n - 1) {
                    return true;
                }
            }
        }
        return false;
    }
}

// 435. 无重叠区间
// https://leetcode.cn/problems/non-overlapping-intervals/solutions/1263171/tan-xin-jie-fa-qi-shi-jiu-shi-yi-ceng-ch-i63h/
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) {
            return 0;
        }

        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[1] - interval2[1];
            }
        });

        int n = intervals.length;
        int right = intervals[0][1];
        int ans = 1;
        for (int i = 1; i < n; ++i) {
            if (intervals[i][0] >= right) {
                ++ans;
                right = intervals[i][1];
            }
        }
        return n - ans;
    }
}

// 621. 任务调度器
// https://leetcode.cn/problems/task-scheduler/solutions/509866/jian-ming-yi-dong-de-javajie-da-by-lan-s-jfl9/

// 452. 用最少数量的箭引爆气球
// https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/solutions/2539356/java-tan-xin-tu-jie-yi-dong-by-cao-yang-yjv4c/
class Solution {
    public int findMinArrowShots(int[][] points) {
        // 贪心
        int n = points.length;
        if(n == 0) return 0;

        Arrays.sort(points, (a, b) -> Long.compare(a[1], b[1]));
        int result = 1;
        // 第一支箭直接射出
        int arrow = points[0][1];
        for(int i = 1; i < n; i++){
            if(points[i][0] <= arrow){
                // 该区间能被当前箭right穿过
                continue;
            }
            arrow = points[i][1]; // 继续射出箭
            result++; // 箭数加1
        }
        return result;
    }
}
```

## 字典树：LeetCode820、208、648

```java
// 820. 单词的压缩编码
// https://leetcode.cn/problems/short-encoding-of-words/solutions/174404/99-java-trie-tu-xie-gong-lue-bao-jiao-bao-hui-by-s/
class Solution {
    public int minimumLengthEncoding(String[] words) {
        int len = 0;
        Trie trie = new Trie();
        // 先对单词列表根据单词长度由长到短排序
        Arrays.sort(words, (s1, s2) -> s2.length() - s1.length());
        // 单词插入trie，返回该单词增加的编码长度
        for (String word: words) {
            len += trie.insert(word);
        }
        return len;
    }
}

// 定义tire
class Trie {

    TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public int insert(String word) {
        TrieNode cur = root;
        boolean isNew = false;
        // 倒着插入单词
        for (int i = word.length() - 1; i >= 0; i--) {
            int c = word.charAt(i) - 'a';
            if (cur.children[c] == null) {
                isNew = true; // 是新单词
                cur.children[c] = new TrieNode();
            }
            cur = cur.children[c];
        }
        // 如果是新单词的话编码长度增加新单词的长度+1，否则不变。
        return isNew? word.length() + 1: 0;
    }
}

class TrieNode {
    char val;
    TrieNode[] children = new TrieNode[26];

    public TrieNode() {}
}

// 208. 实现 Trie (前缀树)
class Trie {

    private Node root;

    public Trie() {
        this.root = new Node();
    }

    public void insert(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            int c = word.charAt(i) - 'a';
            if (cur.children[c] == null) {
                cur.children[c] = new Node();
            }
            cur = cur.children[c];
        }
        cur.isEnd = true;
    }

    public boolean search(String word) {
        Node node = searchPrefix(word);
        return node != null && node.isEnd == true;
    }

    public boolean startsWith(String prefix) {
        Node node = searchPrefix(prefix);
        return node != null;

    }

    private Node searchPrefix(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (cur.children[index] == null) {
                return null;
            }
            cur = cur.children[index];
        }
        return cur;
    }
}

class Node {

    Node[] children;
    boolean isEnd;

    public Node() {
        children = new Node[26];
        isEnd = false;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */

// 648. 单词替换
class Solution {
    public String replaceWords(List<String> dictionary, String sentence) {
        Trie trie = new Trie();
        for (String word: dictionary) {
            trie.insert(word);
        }

        String[] words = sentence.split(" ");
        StringBuilder result = new StringBuilder();
        for (String w: words) {
            result.append(trie.search(w)).append(" ");
        }
        return result.substring(0, result.length() - 1);
    }
}

class Trie {

    private Node root;

    public Trie() {
        this.root = new Node();
    }

    public void insert(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            int c = word.charAt(i) - 'a';
            if (cur.children[c] == null) {
                cur.children[c] = new Node();
            }
            cur = cur.children[c];
        }
        cur.isEnd = true;
    }

    public String search(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (cur.children[index] == null) {
                break;
            }
            if (cur.children[index].isEnd) {
                return word.substring(0, i + 1);
            }
            cur = cur.children[index];
        }
        return word;
    }
}

class Node {

    Node[] children;
    boolean isEnd;

    public Node() {
        children = new Node[26];
        isEnd = false;
    }
}
```
