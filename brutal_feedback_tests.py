"""
Comprehensive test suite for brutal feedback optimization.
Loads pre-trained model from brutal_feedback_llm_judge.json.
Focus on demonstrating DSPy's improvement with extensive test cases.
"""

import dspy
import os
from brutal_feedback import (
    BrutalFeedbackModule,
    simple_metric
)

# Use single model
model = dspy.LM(
    model='ollama/phi3:mini',
    api_base='http://localhost:11434',
    api_key='',
    temperature=0.7
)

dspy.configure(lm=model)


# Extensive test cases
COMPREHENSIVE_TESTS = [
    # Code quality
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

    # Architecture decisions
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

    # Business/product ideas
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

    # Security
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

    # Testing
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

    # Documentation
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

    # Performance
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

    # Code style
    {
        "content": "def a(b,c,d,e,f,g,h):\n x=b+c\n y=d+e\n return x*y*f*g*h",
        "context": "Function review",
        "category": "style"
    },

    # Error handling
    {
        "content": "result = api_call()\nif result:\n    return result",
        "context": "Error handling?",
        "category": "errors"
    },
]


def train_and_test():
    """Load pre-trained model and run comprehensive tests."""

    CACHE_FILE = "brutal_feedback_llm_judge.json"

    print("="*80)
    print("LOADING PRE-TRAINED MODEL")
    print("="*80)

    unoptimized = BrutalFeedbackModule()

    if not os.path.exists(CACHE_FILE):
        print(f"\nERROR: Could not find pre-trained model at {CACHE_FILE}")
        print("Please run brutal_feedback.py first to train the model.")
        return

    print(f"\nLoading optimized model from {CACHE_FILE}...")
    optimized = BrutalFeedbackModule()
    optimized.load(CACHE_FILE)
    print("Model loaded successfully!")

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
        unopt_result = unoptimized(content=test['content'], context=test['context'])
        unopt_score = simple_metric(
            dspy.Example(content=test['content'], context=test['context']),
            unopt_result
        )

        # Optimized
        opt_result = optimized(content=test['content'], context=test['context'])
        opt_score = simple_metric(
            dspy.Example(content=test['content'], context=test['context']),
            opt_result
        )

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
