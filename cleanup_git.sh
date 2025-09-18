#!/bin/bash

# Script to remove firebase-config.json from git history
# Run this script to clean up the sensitive data from your repository

echo "Cleaning up firebase-config.json from git history..."

# Remove the file from git tracking
git rm --cached firebase-config.json

# Add the updated .gitignore
git add .gitignore

# Commit the changes
git commit -m "Remove firebase-config.json from tracking and update .gitignore"

echo "✅ firebase-config.json has been removed from git tracking"
echo "✅ .gitignore has been updated to prevent future commits of sensitive files"
echo ""
echo "Now you can push your changes safely:"
echo "git push origin master"
echo ""
echo "⚠️  IMPORTANT: Make sure to keep your firebase-config.json file locally for the application to work!"
