# Fast strafing functions

This repo provides implementations of optimal strafing.  To accelerate
calculations we also vectorised the calculations using SSE4.  These functions
will be useful for simulation purposes, such as for calculating the optimal
path.  For these applications the majority of computing time may be spent on
generating new velocities and positions.

# Usage

To use the strafing functions, include ``strafe-sse4.hpp``.  Suppose we want to
leftstrafe with Half-Life default settings at 1000 fps.  It can be shown that
the optimal angle between velocity and unit acceleration vector in this case is
given by the inverse cosine of a constant divided by speed, which is classified
as *case 2 strafing*.  The functions implementing this have ``c2`` in their
names.

Suppose we want to strafe 1000 frames.  We would start by using
``strafe_c2_precom`` to precompute some constants which will be stored in
``c2_params_t``.

```cpp
c2_params_t c2params;
strafe_c2_precom(30, 0.001 * 320 * 10, c2params);
```

This function needs to be called only once unless one of the settings (such as
frame rate) is changed.  Now we need two arrays of ``N`` elements of
``__m128d`` called ``cts`` and ``sts``.  The first element of each array is
used by the first frame, the second element is used by the second frame, and so
on.  The user can freely choose the array size, though it must be even and
should not be too large so that both arrays can fit inside the cache.  In this
example we use a size of 20.  We also need a variable of ``__m128d`` type to
hold the velocity, where the first element contains the x component of velocity
while the second contains the y component.  In this example the velocity will
be parallel to the x axis with a magnitude of 500.

```cpp
__m128d v = _mm_set_pd(0, 500);
__m128d cts[20], sts[20];
```

Inside the loop we first compute ``cts`` and ``sts`` followed by an inner loop
computing the velocities for 20 frames.

```cpp
for (int i = 0; i < 50; i++) {
  strafe_c2_ctsts(cts, sts, 20, v, c2params);
  for (int j = 0; j < 20; j++)
    v = strafe_c2_side(v, cts[j], sts[j]);
}
```

The usage of case 1 strafing is similar, except we only have one array instead
of two.

Under some situations we can reuse the arrays, as long as the speed and
settings of each frame remain the same.  For example, when optimising the
strafing path we might have the same starting speed and settings, hence the
speeds for subsequent frames will be the same, only with different velocity
directions.  In this case we can pull ``strafe_c2_ctsts`` out of the loop and
invoke it only once, which can significantly reduce the computing time.
