from src.recommendation_data import (
    load_wellness_content,
    load_user_profiles,
    load_user_history,
    get_user_profile,
    get_user_history,
)


def main():
    print("Testing wellness content...")
    wellness_content = load_wellness_content()
    print(f"Wellness items: {len(wellness_content)}")

    print("\nTesting user profiles...")
    profiles = load_user_profiles()
    print(f"User profiles: {len(profiles)}")

    print("\nTesting user history...")
    history = load_user_history()
    print(f"History records: {len(history)}")

    print("\nTesting demo_user_001 profile...")
    profile = get_user_profile("demo_user_001")
    print(f"User ID: {profile['user_id']}")
    print(
        "Preferred categories:",
        ", ".join(profile["preferred_categories"])
    )
    print(
        "Preferred activities:",
        ", ".join(profile["preferred_activity_types"])
    )

    print("\nTesting demo_user_001 history...")
    user_history = get_user_history("demo_user_001")
    print(f"User history records: {len(user_history)}")

    print("\nMost recent interaction:")
    if not user_history.empty:
        latest = user_history.iloc[0]
        print(f"Content: {latest['content_id']}")
        print(f"Emotion: {latest['emotion_state']}")
        print(f"Intensity: {latest['intensity_score']}")
        print(f"Helpful rating: {latest['helpfulness_rating']}")

    print("\nRecommendation data layer test completed successfully.")


if __name__ == "__main__":
    main()