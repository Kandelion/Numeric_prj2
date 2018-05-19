#define NUM_PARTICLES_X 64
#define NUM_PARTICLES_Y 64

void calcSpringForce(int x, int y, __global float4 *pos, float4 *result, float rh, float rv, float rd, float SPRING_K);
inline void calcGravityForce(float4 *result, float PARTICLE_MASS, float GRAVITY);
void calcDampingForce(int x, int y, __global float4 *vel, float4 *result, float DAMPING_CONST);

__kernel
void cloth_position(
	__global float4* pos_in, __global float4* pos_out,
	__global float4* vel_in, __global float4* vel_out,
	__local float4* local_data,
	float3 Gravity,
	float ParticleMass,
	float ParticleInvMass,
	float SpringK,
	float RestLengthHoriz,
	float RestLengthVert,
	float RestLengthDiag,
	float DeltaT,
	float DampingConst) {
	int idx = get_global_id(0) + get_global_size(0) * get_global_id(1);

	int y = idx / NUM_PARTICLES_X;
	int x = idx % NUM_PARTICLES_X;

	float4 F;

	//고정된 점
	if (y == NUM_PARTICLES_Y - 1 && ((x == NUM_PARTICLES_X - 1) || x % (NUM_PARTICLES_X / 4) == 0))
	{
		pos_out[idx] = pos_in[idx];
		return;
	}

	////CalcForce
	F.x = 0; F.y = 0; F.z = 0;
	calcSpringForce(x, y, pos_in, &F, RestLengthHoriz, RestLengthVert, RestLengthDiag, SpringK);
	calcGravityForce(&F, ParticleMass, Gravity.y);
	calcDampingForce(x, y, vel_in, &F, DampingConst);

	////wind
	///*if (flag == 1 && y > NUM_PARTICLES_Y * 0.1)
	//{
	//	F[2] += 15.0f;
	//}*/

	//Apply force
	vel_out[idx] += F * ParticleInvMass * DeltaT;

	//Apply force
	vel_in[idx] = vel_out[idx];

	//x, y, z, padding
	pos_out[idx] = pos_in[idx] + vel_out[idx] * DeltaT;
}

void calcSpringForce(int x, int y, __global float4 *pos, float4 *result, float rh, float rv, float rd, float SPRING_K) {
	float4 r;
	float abs_r = 0.0f;
	float distance;
	int idx0, idx1;

	idx1 = y*NUM_PARTICLES_X + x;

	for (int i = y - 1; i <= y + 1; i++) {
		for (int j = x - 1; j <= x + 1; j++) {
			if (i<0 || j<0 || i >= NUM_PARTICLES_Y || j >= NUM_PARTICLES_X || (i == y && j == x))
				continue;

			idx0 = i*NUM_PARTICLES_X + j;
			r = pos[idx0] - pos[idx1];
			distance = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

			if (i == y) {
				*result += SPRING_K * (distance - rh) * (r / distance);
			}
			else if (j == x) {
				*result += SPRING_K * (distance - rv) * (r / distance);
			}
			else {
				*result += SPRING_K * (distance - rd) * (r / distance);
			}
		}
	}
}

inline void calcGravityForce(float4 *result, float PARTICLE_MASS, float GRAVITY) {
	result->y += PARTICLE_MASS * GRAVITY;
}

void calcDampingForce(int x, int y, __global float4 *vel, float4 *result, float DAMPING_CONST) {
	(*result) -= DAMPING_CONST * vel[x + NUM_PARTICLES_X * y];
}