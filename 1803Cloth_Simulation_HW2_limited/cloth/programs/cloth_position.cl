#define NUM_PARTICLES_X 64
#define NUM_PARTICLES_Y 64

#define GPU 6			//1 : First order, 2 : Cookbook, 3 : Runge Kutta
						//4 : Local_First, 5 : Local_Cook, 6 : Local_Runge

void calcSpringForce(int x, int y, __global float4 *pos, float4 *result, float rh, float rv, float rd, float SPRING_K);
void calcSpringForce_loc(int x, int y, __local float4 *pos, float4 *result, float rh, float rv, float rd, float SPRING_K);
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
#if GPU < 4
	int idx = get_global_id(0) + get_global_size(0) * get_global_id(1);

	int y = idx / NUM_PARTICLES_X;
	int x = idx % NUM_PARTICLES_X;

	float4 F, pos_mid, vel_mid, acc_mid;

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

#if GPU == 1
	//Apply force
	vel_out[idx] += F * ParticleInvMass * DeltaT;
	//Apply force
	vel_in[idx] = vel_out[idx];
	//x, y, z, padding
	pos_out[idx] = pos_in[idx] + vel_out[idx] * DeltaT;
#endif

#if GPU == 2
	//Apply force
	vel_out[idx] += F * ParticleInvMass * DeltaT;
	//Apply force
	vel_in[idx] = vel_out[idx];
	//x, y, z, padding
	pos_out[idx] = pos_in[idx] + vel_in[idx] * DeltaT + 0.5f * F * ParticleInvMass * DeltaT * DeltaT;
#endif

#if GPU == 3
	vel_mid = vel_in[idx] + F * ParticleInvMass * DeltaT;
	pos_mid = pos_in[idx] + vel_mid * DeltaT;
	acc_mid = F * ParticleInvMass;

	pos_out[idx] = pos_mid;
	vel_out[idx] = vel_mid;

	barrier(CLK_GLOBAL_MEM_FENCE);

	//CalcForce
	F.x = 0; F.y = 0; F.z = 0;
	calcSpringForce(x, y, pos_out, &F, RestLengthHoriz, RestLengthVert, RestLengthDiag, SpringK);
	calcGravityForce(&F, ParticleMass, Gravity.y);
	calcDampingForce(x, y, vel_out, &F, DampingConst);

	acc_mid = 0.5f * (acc_mid + F * ParticleInvMass);
	vel_mid = 0.5f * (vel_mid + vel_out[idx]);

	vel_out[idx] = vel_in[idx] + acc_mid * DeltaT;
	pos_out[idx] = pos_in[idx] + vel_mid * DeltaT;

	//Apply force
	vel_in[idx] = vel_out[idx];
#endif
#endif

#if GPU >= 4
	int idx = get_global_id(0) + get_global_size(0) * get_global_id(1);
	int idx_loc = (get_local_id(1) + 1) * (get_local_size(0) + 2) + (get_local_id(0) + 1);

	// Copy into local memory
	uint local_width = get_local_size(0) + 2;

	local_data[(get_local_id(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx];

	// Bottom edge
	if (get_local_id(1) == 0)
	{
		if (get_global_id(1) > 0)
		{
			local_data[get_local_id(0) + 1] = pos_in[idx - get_global_size(0)];

			// Lower left corner
			if (get_local_id(0) == 0)
			{
				if (get_global_id(0) > 0)
				{
					local_data[0] = pos_in[idx - get_global_size(0) - 1];
				}
				else
				{
					local_data[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}

			// Lower right corner
			if (get_local_id(0) == get_local_size(0) - 1)
			{
				if (get_global_id(0) < get_global_size(0) - 1)
				{
					local_data[get_local_size(0) + 1] = pos_in[idx - get_global_size(0) + 1];
				}
				else
				{
					local_data[get_local_size(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
		else
		{
			local_data[get_local_id(0) + 1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Top edge
	if (get_local_id(1) == get_local_size(1) - 1)
	{
		if (get_global_id(1) < get_global_size(1) - 1)
		{
			local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = pos_in[idx + get_global_size(0)];

			// Upper left corner
			if (get_local_id(0) == 0)
			{
				if (get_global_id(0) > 0)
				{
					local_data[(get_local_size(1) + 1) * local_width] = pos_in[idx + get_global_size(0) - 1];
				}
				else
				{
					local_data[(get_local_size(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}

			//Lower right corner
			if (get_local_id(0) == get_local_size(0) - 1)
			{
				if (get_global_id(0) < get_global_size(0) - 1)
				{
					local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + get_global_size(0) + 1];
				}
				else
				{
					local_data[(get_local_size(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
		else
		{
			local_data[(get_local_size(1) + 1) * local_width + (get_local_id(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Left edge
	if (get_local_id(0) == 0)
	{
		if (get_global_id(0) > 0)
		{
			local_data[(get_local_id(1) + 1) * local_width] = pos_in[idx - 1];
		}
		else
		{
			local_data[(get_local_id(1) + 1) * local_width] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

	// Right edge
	if (get_local_id(0) == get_local_size(0) - 1)
	{
		if (get_global_id(0) < get_global_size(0) - 1)
		{
			local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = pos_in[idx + 1];
		}
		else
		{
			local_data[(get_local_id(1) + 1) * local_width + (get_local_size(0) + 1)] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}
/////////////////////////////////////////////////////////////

	barrier(CLK_LOCAL_MEM_FENCE);

/////////////////////////////////////////////////////////////

	int y = idx / NUM_PARTICLES_X;
	int x = idx % NUM_PARTICLES_X;

	float4 F, pos_mid, vel_mid, acc_mid;

	//고정된 점
	if (y == NUM_PARTICLES_Y - 1 && ((x == NUM_PARTICLES_X - 1) || x % (NUM_PARTICLES_X / 4) == 0))
	{
		pos_out[idx] = local_data[idx_loc];
		return;
	}

	//CalcForce
	F.x = 0; F.y = 0; F.z = 0;
	calcSpringForce_loc(x, y, local_data, &F, RestLengthHoriz, RestLengthVert, RestLengthDiag, SpringK);
	calcGravityForce(&F, ParticleMass, Gravity.y);
	calcDampingForce(x, y, vel_in, &F, DampingConst);

#if GPU == 4
	//Apply force
	vel_out[idx] += F * ParticleInvMass * DeltaT;
	//Apply force
	vel_in[idx] = vel_out[idx];
	//x, y, z, padding
	pos_out[idx] = local_data[idx_loc] + vel_out[idx] * DeltaT;
#endif

#if GPU == 5
	//Apply force
	vel_out[idx] += F * ParticleInvMass * DeltaT;
	//Apply force
	vel_in[idx] = vel_out[idx];
	//x, y, z, padding
	pos_out[idx] = local_data[idx_loc] + vel_in[idx] * DeltaT + 0.5f * F * ParticleInvMass * DeltaT * DeltaT;
#endif

#if GPU == 6
	vel_mid = vel_in[idx] + F * ParticleInvMass * DeltaT;
	pos_mid = local_data[idx_loc] + vel_mid * DeltaT;
	acc_mid = F * ParticleInvMass;

	pos_out[idx] = pos_mid;
	vel_out[idx] = vel_mid;

	barrier(CLK_GLOBAL_MEM_FENCE);

	//CalcForce
	F.x = 0; F.y = 0; F.z = 0;
	calcSpringForce_loc(x, y, local_data, &F, RestLengthHoriz, RestLengthVert, RestLengthDiag, SpringK);
	calcGravityForce(&F, ParticleMass, Gravity.y);
	calcDampingForce(x, y, vel_in, &F, DampingConst);

	acc_mid = 0.5f * (acc_mid + F * ParticleInvMass);
	vel_mid = 0.5f * (vel_mid + vel_out[idx]);

	vel_out[idx] = vel_in[idx] + acc_mid * DeltaT;
	pos_out[idx] = local_data[idx_loc] + vel_mid * DeltaT;

	//Apply force
	vel_in[idx] = vel_out[idx];
#endif
#endif
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


void calcSpringForce_loc(int x, int y, __local float4 *pos, float4 *result, float rh, float rv, float rd, float SPRING_K) {
	float4 r;
	float abs_r = 0.0f;
	float distance;
	int idx0, idx1;

	//idx1 = y*NUM_PARTICLES_X + x;
	idx1 = (get_local_id(1) + 1) * (get_local_size(0) + 2) + (get_local_id(0) + 1);

	for (int i = y - 1; i <= y + 1; i++) {
		for (int j = x - 1; j <= x + 1; j++) {
			if (i<0 || j<0 || i >= NUM_PARTICLES_Y || j >= NUM_PARTICLES_X || (i == y && j == x))
				continue;

			//idx0 = i*NUM_PARTICLES_X + j;
			idx0 = (get_local_id(1) + 1 + (i - y)) * (get_local_size(0) + 2) + (get_local_id(0) + 1 + (j - x));
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