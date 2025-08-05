# genie_agent/tools.py

def score_sql_query(sql_query: str) -> float:
  """
  Analyzes a SQL query and returns a complexity score.
  Higher scores indicate more complex queries.

  Args:
    sql_query (str): The SQL query string to analyze.

  Returns:
    float: The complexity score of the query.
  """
  score = 5.0
  if "JOIN" in sql_query.upper():
    score += 10.0
  if "GROUP BY" in sql_query.upper():
    score += 5.0
  if "PARTITION BY" in sql_query.upper():
    score += 10.0
  # Add more rules here for a more sophisticated scorer
  return score

def review_complexity_score(score: float) -> str:
  """
  Reviews a complexity score and returns a decision.

  Args:
    score (float): The complexity score to review.

  Returns:
    str: The decision, either "approved" or "too_complex".
  """
  # The complexity threshold is set to 10.0
  if score > 10.0:
    return "too_complex"
  return "approved"
