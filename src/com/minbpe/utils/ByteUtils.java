package utils;

/**
 * Utility class for byte array operations.
 */
public class ByteUtils {
    /**
     * Concatenates two byte arrays.
     *
     * @param a The first byte array.
     * @param b The second byte array.
     * @return The concatenated byte array.
     */
    public static byte[] concatenate(byte[] a, byte[] b) {
        byte[] result = new byte[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
