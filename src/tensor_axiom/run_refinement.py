#!/usr/bin/env python3
"""
Main script to run axiom refinement loop

Usage:
    python -m tensor_axiom.run_refinement [command]

Commands:
    test     - Run test cycle
    review   - Display review queue
    inspect  - Inspect specific axiom
    approve  - Approve axiom
    reject   - Reject axiom
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensor_axiom.refinement_loop import RefinementLoop


def main():
    parser = argparse.ArgumentParser(description="Axiom Refinement System")
    parser.add_argument('command', choices=['test', 'review', 'inspect', 'approve', 'reject', 'stats'],
                       help='Command to execute')
    parser.add_argument('--axiom-id', type=str, help='Axiom ID for inspect/approve/reject')
    parser.add_argument('--notes', type=str, default='', help='Notes for approve/reject')
    parser.add_argument('--limit', type=int, default=10, help='Limit for review display')
    
    args = parser.parse_args()
    
    # Initialize refinement loop
    loop = RefinementLoop()
    
    if args.command == 'test':
        print("\nRunning test cycle...")
        loop.run_test_cycle()
        print("\n✓ Test cycle complete")
        
    elif args.command == 'review':
        loop.display_review_queue(limit=args.limit)
        
    elif args.command == 'inspect':
        if not args.axiom_id:
            print("Error: --axiom-id required for inspect command")
            sys.exit(1)
        loop.review_axiom(args.axiom_id)
        
    elif args.command == 'approve':
        if not args.axiom_id:
            print("Error: --axiom-id required for approve command")
            sys.exit(1)
        loop.approve_axiom(args.axiom_id, args.notes)
        
    elif args.command == 'reject':
        if not args.axiom_id:
            print("Error: --axiom-id required for reject command")
            sys.exit(1)
        loop.reject_axiom(args.axiom_id, args.notes)
        
    elif args.command == 'stats':
        stats = loop.library.get_stats()
        print("\n" + "=" * 60)
        print("AXIOM LIBRARY STATISTICS")
        print("=" * 60)
        print(f"\nTotal axioms: {stats['total_axioms']}")
        print(f"Tested: {stats['tested_axioms']}")
        print(f"Untested: {stats['untested_axioms']}")
        print(f"Average success rate: {stats['average_success_rate']:.1%}")
        print(f"Needs review: {stats['needs_review']}")
        print(f"\nCategories:")
        for cat, count in stats['categories'].items():
            print(f"  {cat}: {count}")


if __name__ == '__main__':
    main()
