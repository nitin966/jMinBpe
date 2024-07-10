package utils;

import java.nio.charset.StandardCharsets;

/**
 * Utility class for token-related operations.
 */
public class TokenUtils {
    /**
     * Renders a token for human-readable output, escaping control characters.
     *
     * @param token The token to render.
     * @return A string representation of the token.
     */
    public static String renderToken(byte[] token) {
        String s = new String(token, StandardCharsets.UTF_8);
        return replaceControlCharacters(s);
    }

    /**
     * Replaces control characters in a string with their Unicode escape sequences.
     *
     * @param s The input string.
     * @return The string with control characters replaced.
     */
    public static String replaceControlCharacters(String s) {
        StringBuilder result = new StringBuilder(s.length());
        for (char ch : s.toCharArray()) {
            if (Character.getType(ch) != Character.CONTROL) {
                result.append(ch);
            } else {
                result.append(String.format("\\u%04x", (int) ch));
            }
        }
        return result.toString();
    }
}
