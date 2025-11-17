"""
Comprehensive test suite for brutal feedback optimization.
Uses same model for generation and evaluation (simplified).
Focus on demonstrating DSPy's improvement with extensive test cases.
"""

import dspy
from brutal_feedback import (
    FeedbackGenerator, BrutalFeedbackModule,
    TRAINING_EXAMPLES, create_dspy_examples
)

# Use single model
model = dspy.LM(
    model='ollama/qwen2.5:0.5b',
    api_base='http://localhost:11434',
    api_key='',
    temperature=2
)

dspy.configure(lm=model)


# Extensive test cases
COMPREHENSIVE_TESTS = [
    # Code quality (15 tests)
    {
        "content": "if x == True:\n    return True\nelse:\n    return False",
        "context": "What's wrong here?",
        "category": "code_smell"
    },
    {
        "content": "def process(data):\n    # TODO: implement\n    pass",
        "context": "Review this function",
        "category": "incomplete"
    },
    {
        "content": "import *\nfrom module import *",
        "context": "Imports review",
        "category": "bad_practice"
    },
    {
        "content": "x = 1\ny = 2\nz = 3\na = 4\nb = 5",
        "context": "Variable naming",
        "category": "naming"
    },
    {
        "content": "def calculate(a, b, c, d, e, f, g, h, i, j):\n    return a + b + c + d + e + f + g + h + i + j",
        "context": "Function signature review",
        "category": "code_smell"
    },
    {
        "content": "# God object with 50 methods\nclass Manager:\n    def do_everything(self): pass",
        "context": "Class design",
        "category": "bad_practice"
    },
    {
        "content": "global_state = {}\ndef update():\n    global global_state\n    global_state['x'] = 5",
        "context": "State management",
        "category": "code_smell"
    },
    {
        "content": "result = eval(user_input)",
        "context": "User input handling",
        "category": "security"
    },
    {
        "content": "while True:\n    try:\n        break\n    except:\n        continue",
        "context": "Loop logic",
        "category": "code_smell"
    },
    {
        "content": "x = [1,2,3,4,5]\ny = [1,2,3,4,5]\nz = [1,2,3,4,5]",
        "context": "Duplication review",
        "category": "bad_practice"
    },
    {
        "content": "if (((x > 5) and (y < 10)) or ((z == 3) and (a != 7))):\n    return True",
        "context": "Conditional logic",
        "category": "code_smell"
    },
    {
        "content": "def func():\n    return func()",
        "context": "Recursion check",
        "category": "bad_practice"
    },
    {
        "content": "list = [1, 2, 3]\ndict = {'a': 1}",
        "context": "Variable names",
        "category": "naming"
    },
    {
        "content": "def f(x):\n    return x if x else None",
        "context": "Function naming",
        "category": "naming"
    },
    {
        "content": "class data:\n    pass",
        "context": "Class naming",
        "category": "naming"
    },

    # Architecture decisions (10 tests)
    {
        "content": "Let's use MongoDB because it's web scale",
        "context": "Database choice",
        "category": "architecture"
    },
    {
        "content": "We'll handle 1M users by adding more servers",
        "context": "Scalability plan",
        "category": "scaling"
    },
    {
        "content": "Let's rewrite in Rust for performance",
        "context": "Should we?",
        "category": "rewrite"
    },
    {
        "content": "We need Kubernetes for our 3-person startup",
        "context": "Infrastructure decision",
        "category": "over_engineering"
    },
    {
        "content": "NoSQL is always faster than SQL",
        "context": "Database opinion",
        "category": "architecture"
    },
    {
        "content": "Let's split this into 20 microservices",
        "context": "Service architecture",
        "category": "over_engineering"
    },
    {
        "content": "We'll use event sourcing for everything",
        "context": "Data architecture",
        "category": "over_engineering"
    },
    {
        "content": "Just throw more RAM at it",
        "context": "Performance issue",
        "category": "scaling"
    },
    {
        "content": "We'll use GraphQL because REST is old",
        "context": "API design",
        "category": "architecture"
    },
    {
        "content": "Serverless will solve all our problems",
        "context": "Infrastructure choice",
        "category": "buzzword"
    },

    # Business/product ideas (10 tests)
    {
        "content": "It's like Uber but for dog walking",
        "context": "Startup pitch",
        "category": "idea"
    },
    {
        "content": "We'll monetize later once we have users",
        "context": "Business model",
        "category": "business"
    },
    {
        "content": "AI will solve all our problems",
        "context": "Product strategy",
        "category": "buzzword"
    },
    {
        "content": "We're building a social network for developers",
        "context": "Product idea",
        "category": "idea"
    },
    {
        "content": "First mover advantage will protect us",
        "context": "Competitive strategy",
        "category": "business"
    },
    {
        "content": "We don't have competitors",
        "context": "Market analysis",
        "category": "business"
    },
    {
        "content": "Users will pay for this because it's better",
        "context": "Monetization",
        "category": "business"
    },
    {
        "content": "Let's pivot to blockchain",
        "context": "Strategy shift",
        "category": "buzzword"
    },
    {
        "content": "We'll be the X of Y",
        "context": "Positioning",
        "category": "idea"
    },
    {
        "content": "Growth hacking will get us users",
        "context": "Marketing strategy",
        "category": "buzzword"
    },

    # Security (8 tests)
    {
        "content": "password = request.GET['pwd']",
        "context": "Authentication code",
        "category": "security"
    },
    {
        "content": "We'll add security later",
        "context": "MVP planning",
        "category": "security_planning"
    },
    {
        "content": "query = 'SELECT * FROM users WHERE id=' + user_id",
        "context": "Database query",
        "category": "security"
    },
    {
        "content": "API_KEY = '12345abcde'",
        "context": "Code review",
        "category": "security"
    },
    {
        "content": "subprocess.call(user_input, shell=True)",
        "context": "Command execution",
        "category": "security"
    },
    {
        "content": "We don't need HTTPS in development",
        "context": "Dev environment",
        "category": "security_planning"
    },
    {
        "content": "Admin password: admin123",
        "context": "Credentials",
        "category": "security"
    },
    {
        "content": "CORS: allow all origins",
        "context": "API configuration",
        "category": "security"
    },

    # Testing (7 tests)
    {
        "content": "# Tests are for people who don't know how to code",
        "context": "Testing philosophy",
        "category": "testing"
    },
    {
        "content": "time.sleep(5) # wait for db",
        "context": "Test code",
        "category": "flaky_tests"
    },
    {
        "content": "# This test passes 90% of the time",
        "context": "Test reliability",
        "category": "flaky_tests"
    },
    {
        "content": "assert True # TODO: write real test",
        "context": "Test implementation",
        "category": "testing"
    },
    {
        "content": "# Coverage is 100% so we're good",
        "context": "Quality metric",
        "category": "testing"
    },
    {
        "content": "Mock everything in tests",
        "context": "Testing strategy",
        "category": "testing"
    },
    {
        "content": "We'll test in production",
        "context": "QA process",
        "category": "testing"
    },

    # Documentation (6 tests)
    {
        "content": "# This function does stuff",
        "context": "Comment review",
        "category": "documentation"
    },
    {
        "content": "README: Coming soon!",
        "context": "Project docs",
        "category": "documentation"
    },
    {
        "content": "// Magic number\nconst X = 42;",
        "context": "Code comment",
        "category": "documentation"
    },
    {
        "content": "Documentation is for users, not developers",
        "context": "Documentation philosophy",
        "category": "documentation"
    },
    {
        "content": "The code is self-documenting",
        "context": "Documentation approach",
        "category": "documentation"
    },
    {
        "content": "# Bug fix\ngit commit -m 'fix'",
        "context": "Commit message",
        "category": "documentation"
    },

    # Performance (7 tests)
    {
        "content": "for i in range(1000000):\n    list.append(expensive_operation())",
        "context": "Optimize this",
        "category": "performance"
    },
    {
        "content": "We'll cache everything in memory",
        "context": "Performance strategy",
        "category": "caching"
    },
    {
        "content": "SELECT * FROM huge_table",
        "context": "Database query",
        "category": "performance"
    },
    {
        "content": "Premature optimization is evil, so we never optimize",
        "context": "Performance philosophy",
        "category": "performance"
    },
    {
        "content": "for row in db.query():\n    for item in api.fetch():\n        process()",
        "context": "Nested loops",
        "category": "performance"
    },
    {
        "content": "Loading all 10GB into memory for processing",
        "context": "Data processing",
        "category": "performance"
    },
    {
        "content": "We'll add indexes if it gets slow",
        "context": "Database design",
        "category": "performance"
    },

    # Error handling (7 tests)
    {
        "content": "result = api_call()\nif result:\n    return result",
        "context": "Error handling?",
        "category": "errors"
    },
    {
        "content": "try:\n    risky_thing()\nexcept:\n    pass",
        "context": "Exception handling",
        "category": "errors"
    },
    {
        "content": "It works on my machine",
        "context": "Bug report response",
        "category": "errors"
    },
    {
        "content": "except Exception as e:\n    print(e)",
        "context": "Error handling",
        "category": "errors"
    },
    {
        "content": "if error:\n    return None",
        "context": "Error handling pattern",
        "category": "errors"
    },
    {
        "content": "We'll log errors to /dev/null",
        "context": "Logging strategy",
        "category": "errors"
    },
    {
        "content": "assert False, 'This should never happen'",
        "context": "Defensive programming",
        "category": "errors"
    },

    # Code style (5 tests)
    {
        "content": "def a(b,c,d,e,f,g,h):\n x=b+c\n y=d+e\n return x*y*f*g*h",
        "context": "Function review",
        "category": "style"
    },
    {
        "content": "if(x==5){return true;}",
        "context": "Python code review",
        "category": "style"
    },
    {
        "content": "def myFunction(): pass\ndef my_other_function(): pass",
        "context": "Naming consistency",
        "category": "style"
    },
    {
        "content": "x=1;y=2;z=3;return x+y+z",
        "context": "Code formatting",
        "category": "style"
    },
    {
        "content": "VeryLongVariableNameThatDescribesExactlyWhatItDoes = 5",
        "context": "Variable naming",
        "category": "style"
    },

    # Misc/Process (5 tests)
    {
        "content": "We don't need version control for a small project",
        "context": "Project setup",
        "category": "process"
    },
    {
        "content": "Code reviews slow us down",
        "context": "Development process",
        "category": "process"
    },
    {
        "content": "Let's skip staging and deploy to prod",
        "context": "Deployment process",
        "category": "process"
    },
    {
        "content": "Force push to main is fine",
        "context": "Git workflow",
        "category": "process"
    },
    {
        "content": "We'll document the API after launch",
        "context": "API development",
        "category": "process"
    },
]


def simple_metric(example, prediction, trace=None) -> float:
    """Simple deterministic metric based on feedback length and keywords."""
    feedback = prediction.feedback if hasattr(prediction, 'feedback') else str(prediction)

    score = 0.0

    # Must have substantive feedback (15+ words)
    word_count = len(feedback.split())
    if word_count >= 15:
        score += 0.3
    elif word_count >= 8:
        score += 0.15

    # Bonus for honest/direct language
    honest_words = ['wrong', 'bad', 'terrible', 'avoid', 'never', 'stop', 'don\'t', 'no']
    if any(word in feedback.lower() for word in honest_words):
        score += 0.2

    # Bonus for specific suggestions
    action_words = ['use', 'try', 'replace', 'change', 'instead', 'should']
    if any(word in feedback.lower() for word in action_words):
        score += 0.3

    # Bonus for explaining why
    reasoning_words = ['because', 'since', 'will cause', 'leads to', 'results in']
    if any(word in feedback.lower() for word in reasoning_words):
        score += 0.2

    return min(score, 1.0)


def train_and_test():
    """Train model and run comprehensive tests."""

    print("="*80)
    print("TRAINING PHASE")
    print("="*80)

    train_examples = create_dspy_examples(TRAINING_EXAMPLES)
    print(f"Training examples: {len(train_examples)}")

    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(
        metric=simple_metric,
        max_bootstrapped_demos=5,
        max_labeled_demos=5,
        max_rounds=1
    )

    unoptimized = BrutalFeedbackModule()

    print("\nOptimizing...")
    optimized = optimizer.compile(unoptimized, trainset=train_examples)

    print("\n" + "="*80)
    print("COMPREHENSIVE TESTING - {} test cases".format(len(COMPREHENSIVE_TESTS)))
    print("="*80)

    results_by_category = {}

    for i, test in enumerate(COMPREHENSIVE_TESTS, 1):
        category = test['category']
        if category not in results_by_category:
            results_by_category[category] = {'unopt': [], 'opt': []}

        print(f"\n[{i}/{len(COMPREHENSIVE_TESTS)}] {category.upper()}")
        print(f"Content: {test['content'][:70]}...")
        print(f"Context: {test['context']}")

        # Unoptimized
        try:
            unopt_result = unoptimized(content=test['content'], context=test['context'])
            unopt_score = simple_metric(
                dspy.Example(content=test['content'], context=test['context']),
                unopt_result
            )
        except Exception as e:
            print(f"\n  UNOPTIMIZED: Failed with error - {str(e)[:100]}")
            unopt_result = dspy.Prediction(feedback="[Model failed to generate valid output]")
            unopt_score = 0.0

        # Optimized
        try:
            opt_result = optimized(content=test['content'], context=test['context'])
            opt_score = simple_metric(
                dspy.Example(content=test['content'], context=test['context']),
                opt_result
            )
        except Exception as e:
            print(f"\n  OPTIMIZED: Failed with error - {str(e)[:100]}")
            opt_result = dspy.Prediction(feedback="[Model failed to generate valid output]")
            opt_score = 0.0

        results_by_category[category]['unopt'].append(unopt_score)
        results_by_category[category]['opt'].append(opt_score)

        print(f"\n  UNOPTIMIZED (score: {unopt_score:.2f}):")
        print(f"    {unopt_result.feedback}")

        print(f"\n  OPTIMIZED (score: {opt_score:.2f}):")
        print(f"    {opt_result.feedback}")

        improvement = opt_score - unopt_score
        print(f"\n  IMPROVEMENT: {improvement:+.2f}")
        print("-"*80)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY BY CATEGORY")
    print("="*80)

    overall_unopt = []
    overall_opt = []

    for category, scores in sorted(results_by_category.items()):
        avg_unopt = sum(scores['unopt']) / len(scores['unopt'])
        avg_opt = sum(scores['opt']) / len(scores['opt'])
        improvement = avg_opt - avg_unopt

        overall_unopt.extend(scores['unopt'])
        overall_opt.extend(scores['opt'])

        print(f"\n{category}:")
        print(f"  Unoptimized: {avg_unopt:.3f}")
        print(f"  Optimized:   {avg_opt:.3f}")
        print(f"  Improvement: {improvement:+.3f}")

    # Overall
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)

    total_unopt = sum(overall_unopt) / len(overall_unopt)
    total_opt = sum(overall_opt) / len(overall_opt)
    total_improvement = total_opt - total_unopt
    pct_improvement = (total_improvement / total_unopt * 100) if total_unopt > 0 else 0

    print(f"\nAverage Score Across {len(COMPREHENSIVE_TESTS)} Tests:")
    print(f"  Unoptimized: {total_unopt:.3f}")
    print(f"  Optimized:   {total_opt:.3f}")
    print(f"  Improvement: {total_improvement:+.3f} ({pct_improvement:+.1f}%)")

    print(f"\nTests where optimized won: {sum(1 for u, o in zip(overall_unopt, overall_opt) if o > u)}/{len(COMPREHENSIVE_TESTS)}")
    print(f"Tests where scores tied: {sum(1 for u, o in zip(overall_unopt, overall_opt) if o == u)}/{len(COMPREHENSIVE_TESTS)}")
    print(f"Tests where unoptimized won: {sum(1 for u, o in zip(overall_unopt, overall_opt) if o < u)}/{len(COMPREHENSIVE_TESTS)}")


if __name__ == "__main__":
    train_and_test()
